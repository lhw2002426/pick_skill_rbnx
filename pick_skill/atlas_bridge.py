# SPDX-License-Identifier: MulanPSL-2.0
"""pick_skill_rbnx atlas bridge — Capability + contract-typed MCP tool.

Orchestrates the three upstream perception+manipulation services into
a single user-invocable `pick(object_name)` operation, exposed as the
`robonix/skill/pick/pick` MCP tool.

Pipeline (all over MCP HTTP — atlas-resolved endpoints, no hardcoded
ROS topic / service names):

    Pilot LLM  ──pick("the comb")──►  this skill
                                          │
                  ┌────────── 1. detect_object ─────────────►  yolo_world_rbnx
                  │            (object_name → bbox + 3D center)
                  ▼
                  for retry in 0..N:
                      ┌──── 2. grasp_request ────►  yolo_grasp_rbnx
                      │       (name + bbox + center + retry → grasp_pose)
                      │
                      ▼
                      ┌──── 3. execute_grasp ────►  piper_moveit_rbnx
                      │       (grasp_pose + gripper_width + timeout)
                      │
                      ▼
                      success → return
                      failure → next retry (different grasp candidate)

Why MCP-everywhere instead of mixing in raw ROS service calls (the
upstream pick.py path):

* Consistent contract surface — pilot's view of the system is the
  set of robonix/* contracts; mixing ROS services means the LLM sees
  inconsistent capability shapes for "morally equivalent" calls.
* Topology independence — if a future deploy reshuffles which
  package owns which service, atlas resolution adapts; hardcoded
  topic names don't.
* No /arm/arm_status polling here — execute_grasp already does that
  internally (Stage 5 piper_moveit_rbnx) and only returns once the
  arm has hit busy → idle. We just await its response.

Lifecycle (Skill — lazy activate, same as explore_rbnx):
    on_init      — light. State machine reaches INITIALIZED at boot.
                   No atlas resolution here (upstream services may
                   still be warming up).
    on_activate  — heavy. Sent by executor on first MCP call into
                   any of this skill's tools. Resolve the three
                   upstream MCP endpoints, build the FastMCP client,
                   ready to serve.
    on_deactivate — drop the FastMCP client; idle-evicted.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from typing import Any, Optional

from robonix_api import ATLAS, Skill, Ok, Err  # noqa: E402

logging.basicConfig(
    level=os.environ.get("PICK_SKILL_LOG_LEVEL", "INFO"),
    format="[pick_skill] %(message)s",
)
log = logging.getLogger("pick_skill")

# Provider id MUST match the deploy manifest's `skill: - name: ...`.
pick_skill = Skill(
    id=os.environ.get("ROBONIX_CAPABILITY_ID", "pick"),
    namespace="robonix/skill/pick",
)


# ── shared state (between on_activate + handler) ────────────────────────────
_state_lock = threading.Lock()
# Resolved upstream MCP endpoints. None until on_activate completes.
# Three keys, three contract ids — see REQUIRED_INPUTS below.
_endpoints: Optional[dict[str, str]] = None
_default_timeout_s   = 60.0
_default_max_retries = 5

# We deliberately keep ONE FastMCP Client per upstream URL (lazily
# constructed in the handler, not on_activate). FastMCP's Client is
# async-context-manager based: each call_tool happens inside an
# `async with client as c: ...` block. We don't want to build/tear
# the client per-call (3 round-trips per pick = 3× connect/handshake
# overhead), so we cache the URL→Client mapping and reuse.
_mcp_clients_lock = threading.Lock()
_mcp_clients: dict[str, Any] = {}   # base_url → fastmcp.Client


# ── atlas-resolved upstream contracts ───────────────────────────────────────
REQUIRED_INPUTS = {
    "detect_object":  ("robonix/service/perception/object_detect/detect_object", "mcp"),
    "grasp_request":  ("robonix/service/perception/grasp_pose/grasp_request",    "mcp"),
    "execute_grasp":  ("robonix/service/manipulation/execute_grasp",             "mcp"),
}

# Optional upstreams — pick still activates if these are missing on
# atlas, but each key has degraded behaviour documented:
#   "reset" — calls /moveit_control/reset on the cpp executor at
#             the start of every pick() so its sticky state flags
#             (is_busy_, need_to_adjust_gripper_, …) are clean.
#             If unavailable, pick will skip the reset call; the
#             first pick still works, but a second pick is likely
#             to wedge on cpp.is_busy_=true. Fixed in piper_moveit_rbnx
#             >= the commit that added /moveit_control/reset.
OPTIONAL_INPUTS = {
    "reset":          ("robonix/service/manipulation/reset",                     "mcp"),
}


def _resolve_inputs(deadline_s: float = 60.0) -> dict[str, str]:
    """Block until atlas can resolve all REQUIRED_INPUTS upstream MCP
    endpoints, or fail loudly. Then best-effort resolve OPTIONAL_INPUTS
    — missing optionals only generate a warning. Same shape as
    explore_rbnx.resolve_inputs."""
    resolved: dict[str, str] = {}
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        for key, (cid, transport) in REQUIRED_INPUTS.items():
            if key in resolved:
                continue
            try:
                cap_view = ATLAS.find_unique_capability(
                    contract_id=cid, transport=transport)
                ch = pick_skill.connect_capability(cap_view, cid, transport)
            except Exception:  # noqa: BLE001
                continue
            ep = ch.endpoint
            try:
                ch.close()
            except Exception:  # noqa: BLE001
                pass
            if ep:
                resolved[key] = ep
                log.info("resolved %s [%s] → %s", cid, transport, ep)
        if len(resolved) == len(REQUIRED_INPUTS):
            break
        time.sleep(2.0)

    missing = [k for k in REQUIRED_INPUTS if k not in resolved]
    if missing:
        raise RuntimeError(
            f"pick skill cannot find dependencies on atlas: missing "
            f"{[REQUIRED_INPUTS[k][0] for k in missing]}. The skill needs "
            f"yolo_world_rbnx (object_detect) + yolo_grasp_rbnx (grasp_pose) "
            f"+ piper_moveit_rbnx (manipulation/execute_grasp) all ACTIVE "
            f"before it can run. There is intentionally no ROS-service "
            f"fallback — packaging-spec invariant #1.")

    # Best-effort resolve OPTIONAL_INPUTS. Single shot — if they're not
    # advertised yet, log a warning and move on. They'll be silently
    # skipped at call time.
    for key, (cid, transport) in OPTIONAL_INPUTS.items():
        try:
            cap_view = ATLAS.find_unique_capability(
                contract_id=cid, transport=transport)
            ch = pick_skill.connect_capability(cap_view, cid, transport)
            ep = ch.endpoint
            try:
                ch.close()
            except Exception:  # noqa: BLE001
                pass
            if ep:
                resolved[key] = ep
                log.info("resolved %s [%s] → %s (optional)", cid, transport, ep)
        except Exception:  # noqa: BLE001
            log.warning(
                "optional dep %s not on atlas — pick will run without it. "
                "See OPTIONAL_INPUTS docstring for degraded behaviour.", cid)

    return resolved


# ── MCP client helpers ──────────────────────────────────────────────────────
def _mcp_client_for(url: str):
    """Return a (cached) fastmcp.Client for `url`. Lazy import — so
    unit tests / dry-validate don't pull in fastmcp."""
    with _mcp_clients_lock:
        c = _mcp_clients.get(url)
        if c is not None:
            return c
        from fastmcp import Client
        c = Client(url)
        _mcp_clients[url] = c
        return c


async def _mcp_call(url: str, tool: str, args: dict) -> dict:
    """Single MCP tool round-trip. Returns the parsed dict response."""
    client = _mcp_client_for(url)
    async with client as c:
        result = await c.call_tool(tool, args)
        if not result.content:
            return {}
        txt = result.content[0].text
        try:
            return json.loads(txt)
        except Exception:  # noqa: BLE001
            return {"raw": txt}


def _mcp_call_sync(url: str, tool: str, args: dict) -> dict:
    """Sync wrapper; spins a private event loop. We sync-call from the
    @pick_skill.mcp handler thread — that thread isn't asyncio-aware
    so asyncio.run() is fine."""
    try:
        return asyncio.run(_mcp_call(url, tool, args))
    except Exception as e:  # noqa: BLE001
        log.warning("mcp call %s failed: %s", tool, e)
        return {"_error": str(e)}


# ── pipeline stages ─────────────────────────────────────────────────────────
def _stage_reset() -> dict:
    """Stage 0 (preflight): reset the manipulation state machine.

    Calls piper_moveit_rbnx's `manipulation/reset` MCP, which in turn
    calls /moveit_control/reset on the cpp executor. Clears sticky
    is_busy_ flags and parks the arm at init pose so the upcoming
    grasp starts from a known-clean state.

    Optional — if reset capability isn't on atlas (e.g. older
    piper_moveit_rbnx), this is a no-op and pick() proceeds without
    it. Documented degradation: a second pick() in the same session
    may wedge on the cpp's stale is_busy_=true.
    """
    assert _endpoints is not None
    if "reset" not in _endpoints:
        log.warning("stage0 reset SKIPPED — "
                    "manipulation/reset not on atlas (older piper_moveit?)")
        return {"success": True, "message": "skipped (capability missing)",
                "elapsed_s": 0.0}
    log.info("stage0 reset (clear cpp state machine, park arm)")
    resp = _mcp_call_sync(_endpoints["reset"], "reset", {"ack": True})
    log.info("stage0 result: success=%s msg=%r elapsed=%.2fs",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("elapsed_s", 0.0)))
    return resp


def _stage_detect_object(object_name: str) -> dict:
    """Stage 1: localize the object in the camera frame."""
    assert _endpoints is not None
    args = {"object_name": object_name}
    log.info("stage1 detect_object(%r)", object_name)
    resp = _mcp_call_sync(_endpoints["detect_object"], "detect_object", args)
    log.info("stage1 result: success=%s msg=%r conf=%.3f bbox=%s",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("confidence", 0.0)),
             resp.get("bbox_2d"))
    return resp


def _stage_grasp_request(object_name: str, bbox_2d: list, center_3d: list,
                         retry: int) -> dict:
    """Stage 2: compute a grasp pose for the localized object."""
    assert _endpoints is not None
    args = {
        "object_name":      object_name,
        "bbox_2d":          list(bbox_2d) if bbox_2d else [],
        "object_center_3d": list(center_3d) if center_3d else [],
        "retry":            int(retry),
    }
    log.info("stage2 grasp_request retry=%d", retry)
    resp = _mcp_call_sync(_endpoints["grasp_request"], "grasp_request", args)
    log.info("stage2 result: success=%s msg=%r score=%.3f gripper=%.3f",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("score", 0.0)),
             float(resp.get("gripper_width", 0.0)))
    return resp


def _stage_execute_grasp(grasp_pose: dict, gripper_width: float,
                          timeout_s: float) -> dict:
    """Stage 3: drive the arm + close the gripper."""
    assert _endpoints is not None
    args = {
        "target_pose":   grasp_pose,
        "gripper_width": float(gripper_width),
        "timeout_s":     float(timeout_s),
    }
    log.info("stage3 execute_grasp gripper_width=%.3f timeout=%.1fs",
             gripper_width, timeout_s)
    resp = _mcp_call_sync(_endpoints["execute_grasp"], "execute_grasp", args)
    log.info("stage3 result: success=%s msg=%r elapsed=%.2fs",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("elapsed_s", 0.0)))
    return resp


# Default empty PoseStamped dict — used when the response should
# carry "no pose" (e.g. detection failed before any grasp_pose was
# computed). Field names + nesting must match the codegen
# PoseStamped.from_dict() shape (geometry_msgs/PoseStamped).
def _empty_pose_dict() -> dict:
    return {
        "header": {"stamp": {"sec": 0, "nanosec": 0},
                   "frame_id": "camera_color_optical_frame"},
        "pose":   {"position":    {"x": 0.0, "y": 0.0, "z": 0.0},
                   "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
    }


# ── MCP tool (typed against codegen Pick_Request/Pick_Response) ─────────────
from pick_mcp import (  # noqa: E402  pylint: disable=wrong-import-position
    Pick_Request, Pick_Response,
)
# Nested types we need to instantiate when building the response.
from geometry_msgs_mcp import (  # noqa: E402
    PoseStamped, Pose, Point, Quaternion,
)
from std_msgs_mcp import Header  # noqa: E402
from builtin_interfaces_mcp import Time  # noqa: E402


def _build_pose_stamped_from_dict(d: dict) -> PoseStamped:
    """Build the codegen PoseStamped dataclass from an MCP-shape dict.
    The dict structure MUST match what yolo_grasp_rbnx and
    piper_moveit_rbnx return for their PoseStamped fields (their
    to_dict() output)."""
    h_in = d.get("header") or {}
    s_in = h_in.get("stamp") or {}
    p_in = (d.get("pose") or {})
    pos_in = p_in.get("position") or {}
    ori_in = p_in.get("orientation") or {}
    return PoseStamped(
        header=Header(
            stamp=Time(sec=int(s_in.get("sec", 0)),
                       nanosec=int(s_in.get("nanosec", 0))),
            frame_id=str(h_in.get("frame_id", "")),
        ),
        pose=Pose(
            position=Point(
                x=float(pos_in.get("x", 0.0)),
                y=float(pos_in.get("y", 0.0)),
                z=float(pos_in.get("z", 0.0))),
            orientation=Quaternion(
                x=float(ori_in.get("x", 0.0)),
                y=float(ori_in.get("y", 0.0)),
                z=float(ori_in.get("z", 0.0)),
                w=float(ori_in.get("w", 1.0))),
        ),
    )


@pick_skill.mcp("robonix/skill/pick/pick")
def pick(req: Pick_Request) -> Pick_Response:
    """Pick up a specific object using the robot arm.

    Use this tool when the user asks to grasp / pick / fetch / take a
    specific named object visible to the robot's camera. The
    object_name can be any open-vocabulary description (e.g. "red
    cup", "small box", "the comb"). The pipeline detects the object
    with YOLO-World, plans a grasp pose with the geometric yolo_grasp
    estimator, then executes via MoveIt + the Piper arm.

    This call is synchronous — it returns once the arm has either
    completed the grasp or the budget is exhausted. Typical end-to-end
    latency is 10–30 s on a healthy pipeline.

    Returns success=True iff the arm completed the grasp without
    error. On failure, message identifies which stage broke
    (detection / grasp planning / execution / timeout).
    """
    if _endpoints is None:
        return Pick_Response(
            success=False,
            message="pick skill not active (atlas hasn't resolved upstream services yet)",
            grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
            gripper_width=0.0, score=0.0, elapsed_s=0.0,
        )

    object_name  = (req.object_name or "").strip()
    if not object_name:
        return Pick_Response(
            success=False, message="object_name is empty",
            grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
            gripper_width=0.0, score=0.0, elapsed_s=0.0,
        )

    total_to    = float(req.timeout_s) if req.timeout_s > 0 else _default_timeout_s
    max_retries = int(req.max_retries) if req.max_retries > 0 else _default_max_retries

    t0 = time.monotonic()
    deadline = t0 + total_to

    # ── Stage 0: reset (preflight, best-effort) ────────────────────────────
    # Clear the cpp moveit_control state machine + park arm at init.
    # We do NOT abort pick() if reset fails — first picks of a fresh
    # boot don't need it (cpp state is already clean). It only
    # matters for subsequent picks. Failure here just means the
    # next stage will hit the existing wedge if there is one.
    try:
        rst = _stage_reset()
        if not rst.get("success") and "_error" not in rst:
            log.warning("reset returned success=false: %r — "
                        "proceeding anyway", rst.get("message", "")[:80])
    except Exception as e:  # noqa: BLE001
        log.warning("reset stage raised: %s — proceeding anyway", e)

    # ── Stage 1: detect_object ─────────────────────────────────────────────
    det = _stage_detect_object(object_name)
    if "_error" in det or not det.get("success"):
        msg = det.get("_error") or det.get("message", "unknown")
        return Pick_Response(
            success=False, message=f"detection_failed: {msg}",
            grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
            gripper_width=0.0, score=0.0,
            elapsed_s=time.monotonic() - t0,
        )

    bbox_2d   = list(det.get("bbox_2d") or [])
    center_3d = list(det.get("object_center_3d") or [])

    # ── Stage 2 + 3: grasp_request → execute_grasp, retry on failure ─────
    last_grasp_pose_dict: dict = _empty_pose_dict()
    last_gripper_width   = 0.0
    last_score           = 0.0
    last_failure_msg     = "no attempts made"

    for retry in range(max_retries):
        if time.monotonic() >= deadline:
            return Pick_Response(
                success=False,
                message=f"timeout: budget {total_to:.1f}s exhausted before retry {retry}",
                grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
                gripper_width=last_gripper_width, score=last_score,
                elapsed_s=time.monotonic() - t0,
            )

        # Stage 2: grasp_request.
        gr = _stage_grasp_request(object_name, bbox_2d, center_3d, retry)
        if "_error" in gr or not gr.get("success"):
            last_failure_msg = (
                f"grasp_pose retry {retry}: "
                f"{gr.get('_error') or gr.get('message', 'unknown')}")
            log.warning("%s — trying next", last_failure_msg)
            continue

        last_grasp_pose_dict = gr.get("grasp_pose") or _empty_pose_dict()
        last_gripper_width   = float(gr.get("gripper_width", 0.0))
        last_score           = float(gr.get("score", 0.0))

        # Stage 3: execute_grasp. Use the remaining time budget so the
        # whole pick respects req.timeout_s.
        remaining = max(1.0, deadline - time.monotonic())
        # Don't give execute_grasp the entire remaining budget if we
        # still have retries left — leave some headroom for one more
        # grasp_pose attempt if this one fails.
        retries_left = max_retries - retry - 1
        if retries_left > 0:
            exec_to = min(remaining, max(8.0, remaining / (retries_left + 1)))
        else:
            exec_to = remaining

        eg = _stage_execute_grasp(last_grasp_pose_dict, last_gripper_width,
                                   exec_to)

        if "_error" not in eg and eg.get("success"):
            # Success — done.
            return Pick_Response(
                success=True, message="ok",
                grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
                gripper_width=last_gripper_width, score=last_score,
                elapsed_s=time.monotonic() - t0,
            )

        last_failure_msg = (
            f"execute retry {retry}: "
            f"{eg.get('_error') or eg.get('message', 'unknown')}")
        log.warning("%s — trying next grasp candidate", last_failure_msg)

    # Exhausted retries.
    return Pick_Response(
        success=False,
        message=f"all {max_retries} retries failed; last: {last_failure_msg}",
        grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
        gripper_width=last_gripper_width, score=last_score,
        elapsed_s=time.monotonic() - t0,
    )


# ── lifecycle ───────────────────────────────────────────────────────────────
@pick_skill.on_init
def init(cfg):
    """CMD_INIT: light. Don't query atlas — upstream services may
    still be warming up. cfg parsed for forward compat."""
    global _default_timeout_s, _default_max_retries
    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")
    if "default_timeout_s" in cfg:
        try:
            _default_timeout_s = float(cfg["default_timeout_s"])
        except (TypeError, ValueError):
            log.warning("ignoring invalid default_timeout_s: %r",
                        cfg["default_timeout_s"])
    if "default_max_retries" in cfg:
        try:
            _default_max_retries = int(cfg["default_max_retries"])
        except (TypeError, ValueError):
            log.warning("ignoring invalid default_max_retries: %r",
                        cfg["default_max_retries"])
    log.info("CMD_INIT ok (default_timeout_s=%.1f, default_max_retries=%d)",
             _default_timeout_s, _default_max_retries)
    return Ok()


@pick_skill.on_activate
def activate():
    """CMD_ACTIVATE: heavy. Resolve upstream MCP endpoints. Idempotent."""
    global _endpoints
    with _state_lock:
        if _endpoints is not None:
            log.info("CMD_ACTIVATE — already active, no-op")
            return Ok()
        try:
            _endpoints = _resolve_inputs()
        except RuntimeError as e:
            return Err(str(e))
    log.info("CMD_ACTIVATE ok — endpoints resolved: %s", list(_endpoints.keys()))
    return Ok()


@pick_skill.on_deactivate
def deactivate():
    """CMD_DEACTIVATE: drop client cache + endpoints. Idle eviction."""
    global _endpoints
    with _state_lock, _mcp_clients_lock:
        _endpoints = None
        _mcp_clients.clear()
    log.info("CMD_DEACTIVATE ok — endpoints cleared")
    return Ok()


def main() -> int:
    pick_skill.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
