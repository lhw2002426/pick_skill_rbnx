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
                  ┌──── 2. grasp_request ────►  yolo_grasp_rbnx
                  │       (name + bbox + center + retry=0 → grasp_pose)
                      │
                      ▼
                      ┌──── 3. execute_grasp ────►  piper_moveit_rbnx
                      │       (pre_pose + gripper full-open,
                      │        grasp_pose + gripper closed,
                      │        lift + gripper closed)
                      │
                      ▼
                      success → verify gripper did not close past threshold
                                → sleep 2s (visual confirmation)
                                → 4. reset (post-pick park)
                                       ──────►  piper_moveit_rbnx
                                       (open gripper + moveArmtoInit)
                                → return success
                  On timeout, execution failure, or detection failure:
                  also call reset before returning, so cpp's sticky
                  state machine flags (is_busy_, …) don't silently
                  break the NEXT pick.

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
  arm has hit busy → idle. We do subscribe to /arm/joint_states_single
  to verify the gripper did not close completely after the grasp,
  mirroring roboarm's "closed too far means empty" success check.

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
_DEFAULT_TIMEOUT_S   = 60.0
_vla_max_steps       = 50
_gripper_open_width  = 0.08
_gripper_close_width = 0.0
_gripper_grasp_threshold_width = 0.005
_gripper_feedback_timeout_s = 2.0
_gripper_joint_states_topic = "/arm/joint_states_single"
_require_gripper_feedback = True

# Execution mode: "moveit" (traditional pipeline) or "vla" (end-to-end policy)
_mode = "moveit"

# We deliberately keep ONE FastMCP Client per upstream URL (lazily
# constructed in the handler, not on_activate). FastMCP's Client is
# async-context-manager based: each call_tool happens inside an
# `async with client as c: ...` block. We don't want to build/tear
# the client per-call (3 round-trips per pick = 3× connect/handshake
# overhead), so we cache the URL→Client mapping and reuse.
_mcp_clients_lock = threading.Lock()
_mcp_clients: dict[str, Any] = {}   # base_url → fastmcp.Client

_gripper_state_lock = threading.Lock()
_gripper_latest_width: Optional[float] = None
_gripper_latest_seen_mono: Optional[float] = None
_gripper_monitor_node: Any = None
_gripper_monitor_thread: Optional[threading.Thread] = None
_gripper_monitor_stop = threading.Event()


# ── atlas-resolved upstream contracts ───────────────────────────────────────
REQUIRED_INPUTS = {
    "detect_object":  ("robonix/service/perception/object_detect/detect_object", "mcp"),
    "grasp_request":  ("robonix/service/perception/grasp_pose/grasp_request",    "mcp"),
    "execute_grasp":  ("robonix/service/manipulation/execute_grasp",             "mcp"),
}

# Optional upstreams — pick still activates if these are missing on
# atlas, but each key has degraded behaviour documented:
#   "reset" — POST-GRASP park. Called 2s after a successful
#             execute_grasp so the arm parks back at init pose with
#             the gripper open (cpp opens to 0.025 + moveArmtoInit).
#             If unavailable, pick still returns success but the arm
#             stays at the grasp pose with the gripper closed; the
#             NEXT pick will then wedge because cpp's is_busy_=true
#             flag is still set. Fixed in piper_moveit_rbnx >= the
#             commit that added /moveit_control/reset.
#   "demo"  — DEMO mode override. Calls cpp's /moveit_control/demo
#             which opens gripper to 0.08 m, joint-space-moves to
#             the DEMO pose, then closes gripper to 0.025 m
#             (simulating a grasp). Used by the commented-out demo
#             override block at the top of pick() for showcase runs.
#   "demo_place" — sibling of demo for the "place" half of a
#             pick-and-place demo. Calls cpp's
#             /moveit_control/demo_place which drives to DEMO PLACE
#             pose, holds 2 s, opens gripper, parks at init.
OPTIONAL_INPUTS = {
    "reset":          ("robonix/service/manipulation/reset",                     "mcp"),
    "demo":           ("robonix/service/manipulation/demo",                      "mcp"),
    "demo_place":     ("robonix/service/manipulation/demo_place",                "mcp"),
    "vla_execute":    ("robonix/skill/vla/execute",                              "mcp"),
}

# In VLA mode, vla_execute becomes required instead of the traditional pipeline
VLA_REQUIRED_INPUTS = {
    "vla_execute":    ("robonix/skill/vla/execute",                              "mcp"),
}


def _resolve_inputs(deadline_s: float = 60.0) -> dict[str, str]:
    """Block until atlas can resolve all REQUIRED_INPUTS upstream MCP
    endpoints, or fail loudly. Then best-effort resolve OPTIONAL_INPUTS
    — missing optionals only generate a warning. Same shape as
    explore_rbnx.resolve_inputs.

    In VLA mode, only VLA_REQUIRED_INPUTS are mandatory; the traditional
    pipeline deps (detect_object, grasp_request, execute_grasp) become
    optional."""
    # Choose required inputs based on mode
    if _mode == "vla":
        required = VLA_REQUIRED_INPUTS
    else:
        required = REQUIRED_INPUTS

    resolved: dict[str, str] = {}
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        for key, (cid, transport) in required.items():
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
        if len(resolved) == len(required):
            break
        time.sleep(2.0)

    missing = [k for k in required if k not in resolved]
    if missing:
        if _mode == "vla":
            raise RuntimeError(
                f"pick skill (VLA mode) cannot find dependencies on atlas: "
                f"missing {[required[k][0] for k in missing]}. "
                f"vla_client_rbnx must be ACTIVE before pick can run in VLA mode.")
        else:
            raise RuntimeError(
                f"pick skill cannot find dependencies on atlas: missing "
                f"{[required[k][0] for k in missing]}. The skill needs "
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


# ── dedicated background asyncio loop for sync→async bridging ───────────────
# We CAN'T use asyncio.run() from inside the @pick_skill.mcp handler
# because robonix_api's MCP server (fastmcp.streamable_http) calls our
# handler from a thread that already has a running asyncio event loop:
#
#     RuntimeError: asyncio.run() cannot be called from a running event loop
#
# Even when the handler is declared `def` (sync), fastmcp dispatches it
# inside its own loop's executor, which means by the time _mcp_call_sync
# fires there IS a running loop in this thread's context. asyncio.run()
# refuses to nest.
#
# Standard fix: spin up a SEPARATE asyncio loop on a SEPARATE daemon
# thread, and use run_coroutine_threadsafe() to schedule MCP calls onto
# it. The handler thread (which is borrowed from fastmcp) blocks on
# Future.result() — that's just a regular threading wait, not an
# asyncio await, so it doesn't conflict with whatever loop the
# handler thread happens to be running in.
import threading as _threading_for_loop
_bg_loop_lock = _threading_for_loop.Lock()
_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_loop_thread: Optional[_threading_for_loop.Thread] = None


def _ensure_bg_loop() -> asyncio.AbstractEventLoop:
    """Lazily start a daemon thread running its own asyncio loop forever.
    Idempotent — every caller after the first gets the same loop."""
    global _bg_loop, _bg_loop_thread
    with _bg_loop_lock:
        if _bg_loop is not None and _bg_loop.is_running():
            return _bg_loop
        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            finally:
                # Clean exit: drain pending tasks then close.
                try:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True))
                except Exception:  # noqa: BLE001
                    pass
                loop.close()

        t = _threading_for_loop.Thread(
            target=_runner, name="pick-skill-mcp-loop", daemon=True)
        t.start()
        _bg_loop = loop
        _bg_loop_thread = t
        return loop


def _mcp_call_sync(url: str, tool: str, args: dict) -> dict:
    """Sync wrapper for _mcp_call. Schedules the coroutine on a
    dedicated background event loop (see _ensure_bg_loop) and blocks
    on the resulting concurrent.futures.Future. Safe to call from
    inside another asyncio loop's thread (which is what fastmcp does
    when invoking @pick_skill.mcp handlers)."""
    loop = _ensure_bg_loop()
    fut = asyncio.run_coroutine_threadsafe(
        _mcp_call(url, tool, args), loop)
    try:
        # No timeout here — pipeline stages enforce their own budgets
        # via the outer pick timeout budget. If a single MCP RPC
        # hangs forever we'd want a watchdog at a higher layer.
        return fut.result()
    except Exception as e:  # noqa: BLE001
        log.warning("mcp call %s failed: %s", tool, e)
        return {"_error": str(e)}


# ── gripper feedback monitor ───────────────────────────────────────────────
def _joint_state_gripper_width(msg: Any) -> Optional[float]:
    """Extract the actual gripper opening width from JointState.

    piper_ctl publishes /arm/joint_states_single with names
    joint1..joint6, gripper, and position[6] is the Piper gripper
    opening in meters. Accept joint7 too for compatibility with
    MoveIt naming.
    """
    names = list(getattr(msg, "name", []) or [])
    positions = list(getattr(msg, "position", []) or [])
    if not positions:
        return None
    idx: Optional[int] = None
    for candidate in ("gripper", "joint7"):
        if candidate in names:
            idx = names.index(candidate)
            break
    if idx is None and len(positions) >= 7:
        idx = 6
    if idx is None or idx >= len(positions):
        return None
    try:
        width = float(positions[idx])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(width):
        return None
    return max(0.0, width)


def _start_gripper_monitor() -> None:
    """Start a small rclpy subscriber for /arm/joint_states_single.

    The skill remains MCP-first for orchestration, but the gripper
    success signal is only available as ROS JointState feedback today.
    If ROS is unavailable, leave the monitor stopped; the pick path
    will fail at verification time when feedback is required.
    """
    global _gripper_monitor_node, _gripper_monitor_thread
    if _gripper_monitor_thread is not None and _gripper_monitor_thread.is_alive():
        return
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except Exception as e:  # noqa: BLE001
        log.warning("gripper feedback monitor unavailable: %s", e)
        return

    _gripper_monitor_stop.clear()

    class _GripperMonitor(Node):
        def __init__(self) -> None:
            super().__init__("pick_skill_gripper_feedback")
            self.create_subscription(
                JointState, _gripper_joint_states_topic, self._cb, 10)

        def _cb(self, msg: Any) -> None:
            width = _joint_state_gripper_width(msg)
            if width is None:
                return
            with _gripper_state_lock:
                global _gripper_latest_width, _gripper_latest_seen_mono
                _gripper_latest_width = width
                _gripper_latest_seen_mono = time.monotonic()

    def _runner() -> None:
        global _gripper_monitor_node
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            node = _GripperMonitor()
            _gripper_monitor_node = node
            log.info("gripper feedback monitor subscribed to %s",
                     _gripper_joint_states_topic)
            while rclpy.ok() and not _gripper_monitor_stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
            node.destroy_node()
        except Exception as e:  # noqa: BLE001
            log.warning("gripper feedback monitor stopped: %s", e)
        finally:
            _gripper_monitor_node = None

    _gripper_monitor_thread = threading.Thread(
        target=_runner, name="pick-skill-gripper-feedback", daemon=True)
    _gripper_monitor_thread.start()


def _stop_gripper_monitor() -> None:
    global _gripper_monitor_thread
    _gripper_monitor_stop.set()
    thread = _gripper_monitor_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
    _gripper_monitor_thread = None


def _wait_for_gripper_width_after(start_mono: float,
                                  timeout_s: float) -> Optional[float]:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        with _gripper_state_lock:
            width = _gripper_latest_width
            seen = _gripper_latest_seen_mono
        if width is not None and seen is not None and seen >= start_mono:
            return width
        time.sleep(0.05)
    return None


def _verify_gripper_holding(close_started_mono: float) -> tuple[bool, str, float]:
    width = _wait_for_gripper_width_after(
        close_started_mono, _gripper_feedback_timeout_s)
    if width is None:
        msg = (
            f"no gripper feedback on {_gripper_joint_states_topic} within "
            f"{_gripper_feedback_timeout_s:.1f}s"
        )
        if _require_gripper_feedback:
            return False, msg, 0.0
        return True, msg + " (ignored)", 0.0
    if width < _gripper_grasp_threshold_width:
        return (
            False,
            f"gripper closed to {width:.4f}m < "
            f"{_gripper_grasp_threshold_width:.4f}m threshold; likely empty",
            width,
        )
    return (
        True,
        f"gripper held at {width:.4f}m >= "
        f"{_gripper_grasp_threshold_width:.4f}m threshold",
        width,
    )


# ── pipeline stages ─────────────────────────────────────────────────────────
def _stage_reset() -> dict:
    """Post-grasp park: reset the manipulation state machine + park
    the arm at init.

    Calls piper_moveit_rbnx's `manipulation/reset` MCP, which in turn
    calls /moveit_control/reset on the cpp executor. Clears sticky
    is_busy_ flags, opens the gripper to a neutral wide width, and
    parks the arm at init pose so the next pick starts from a
    known-clean state.

    Called BOTH after a successful grasp (with a 2s pre-hold for
    visual confirmation, see pick()) AND after every failed pick
    attempt (to clean up cpp state — most failures leave is_busy_=
    true or a stale need_to_return_init_pose_ flag, which would
    silently break the NEXT pick).

    Optional — if reset capability isn't on atlas (e.g. older
    piper_moveit_rbnx), this is a no-op and pick() still returns
    its grasp result, but the arm will not be parked. The NEXT pick
    will then likely wedge on cpp's stale is_busy_=true flag.
    """
    assert _endpoints is not None
    if "reset" not in _endpoints:
        log.warning("post-grasp reset SKIPPED — "
                    "manipulation/reset not on atlas (older piper_moveit?)")
        return {"success": True, "message": "skipped (capability missing)",
                "elapsed_s": 0.0}
    log.info("post-grasp reset (open gripper, park arm at init)")
    resp = _mcp_call_sync(_endpoints["reset"], "reset", {"ack": True})
    log.info("post-grasp reset result: success=%s msg=%r elapsed=%.2fs",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("elapsed_s", 0.0)))
    return resp


def _stage_demo() -> dict:
    """Demo override: drive the arm to a fixed joint-space DEMO pose
    with the gripper open ~8 cm.

    Calls piper_moveit_rbnx's `manipulation/demo` MCP, which in turn
    calls /moveit_control/demo on the cpp executor. Used by the
    commented-out demo override block at the top of pick() —
    showcase runs replace the real grasp pipeline (yolo_world →
    yolo_grasp → execute_grasp) with this single deterministic call.
    """
    assert _endpoints is not None
    if "demo" not in _endpoints:
        log.warning("demo SKIPPED — manipulation/demo not on atlas "
                    "(older piper_moveit, or demo not yet rebuilt?)")
        return {"success": False, "message": "demo capability not on atlas",
                "elapsed_s": 0.0}
    log.info("demo (open gripper to demo width, move to demo pose)")
    resp = _mcp_call_sync(_endpoints["demo"], "demo", {"ack": True})
    log.info("demo result: success=%s msg=%r elapsed=%.2fs",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("elapsed_s", 0.0)))
    return resp


def _stage_demo_place() -> dict:
    """Demo place override: drive the arm to a fixed joint-space
    DEMO PLACE pose, hold 2 s, open gripper, park at init.

    Calls piper_moveit_rbnx's `manipulation/demo_place` MCP, which in
    turn calls /moveit_control/demo_place on the cpp executor. Used
    by the commented-out demo-place override block in pick() for
    canned pick-and-place showcase runs (paired with _stage_demo).
    """
    assert _endpoints is not None
    if "demo_place" not in _endpoints:
        log.warning("demo_place SKIPPED — manipulation/demo_place not on atlas "
                    "(older piper_moveit, or demo_place not yet rebuilt?)")
        return {"success": False, "message": "demo_place capability not on atlas",
                "elapsed_s": 0.0}
    log.info("demo_place (move to demo place pose, hold, open gripper, park at init)")
    resp = _mcp_call_sync(_endpoints["demo_place"], "demo_place", {"ack": True})
    log.info("demo_place result: success=%s msg=%r elapsed=%.2fs",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("elapsed_s", 0.0)))
    return resp


def _safe_post_pick_reset(context: str) -> None:
    """Best-effort post-pick reset, never raises.

    Used by every pick() exit path (success AND failure) to leave
    cpp in a clean state for the next pick. Logs but swallows any
    exception/error so it can't poison the response we're about to
    return to the caller.
    """
    try:
        rst = _stage_reset()
        if not rst.get("success") and "_error" not in rst:
            log.warning("post-pick reset (%s) returned success=false: %r "
                        "— arm may not be parked at init",
                        context, rst.get("message", "")[:80])
    except Exception as e:  # noqa: BLE001
        log.warning("post-pick reset (%s) raised: %s — continuing", context, e)


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
                   "frame_id": "arm/base_link"},
        "pose":   {"position":    {"x": 0.0, "y": 0.0, "z": 0.0},
                   "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
    }


# ── vertical grasp three-stage motion constants ────────────────────────────
# The vertical-grasp pipeline uses a three-stage "approach → grasp →
# retreat" motion, mirroring roboarm.catch():
#   1. Move to pre_pose (above target, gripper open)  — hover
#   2. Move to grasp_pose (target height, gripper close) — grasp
#   3. Move to pre_pose (above target, gripper close)  — lift
#
# APPROACH_DIST is how far above the grasp pose to hover. This should
# match yolo_grasp_rbnx's config/vertical_grasp.yaml approach_dist.
_APPROACH_DIST = 0.10   # m

# Gripper widths for the three stages.
_GRIPPER_OPEN  = _gripper_open_width
_GRIPPER_CLOSE = _gripper_close_width


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


def _lift_z(pose_dict: dict, dz: float) -> dict:
    """Return a copy of pose_dict with position.z += dz.

    Used to derive the pre/post grasp hover pose from the grasp pose
    in the three-stage vertical-grasp motion.
    """
    import copy
    d = copy.deepcopy(pose_dict)
    pos = d.get("pose", {}).get("position", {})
    pos["z"] = float(pos.get("z", 0.0)) + float(dz)
    return d


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

    # ── DEMO MODE OVERRIDE ────────────────────────────────────────────
    # Uncomment the block below to replace the real grasp pipeline
    # with a single call to manipulation/demo. Every pick.pick(...)
    # then resolves to "open gripper to ~8 cm + move to a fixed
    # joint-space demo pose" — perfect for showcase / canned demos
    # where the agent should look like it's grasping but you don't
    # want to depend on perception + grasp planning succeeding.
    #
    # The demo path returns success=True unconditionally (i.e. it
    # surfaces success=True to pilot whether or not the underlying
    # cpp /moveit_control/demo Trigger reported success), so the
    # LLM never says "I tried to grasp but failed" during a demo.
    # If the cpp side genuinely fails, the failure is logged loudly
    # but pick() still returns success=True.
    #
    # Mirroring the real-grasp success path, after demo lands we
    # also sleep 2 s for visual confirmation and then call
    # _safe_post_pick_reset() so the gripper closes back to 0.025
    # and the arm parks at init (cpp's resetCallback) — without
    # that step the demo would leave the arm stuck at the demo pose
    # with the gripper open, and the NEXT pick would wedge on
    # cpp's stale is_busy_ flag.
    #
    # To enable for a demo:
    #   1. Make sure piper_moveit_rbnx is rebuilt with the demo
    #      service (commit adding /moveit_control/demo).
    #   2. Uncomment the entire `if True:` block below + its return.
    #   3. Restart pick_skill_rbnx (or just send CMD_DEACTIVATE +
    #      CMD_ACTIVATE so OPTIONAL_INPUTS gets re-resolved with
    #      the new demo endpoint).
    # To disable: re-comment.
    #
    # ── BEGIN demo override ──
    # if True:
    #     log.info("DEMO MODE: bypassing real grasp pipeline; "
    #              "calling manipulation/demo for object_name=%r",
    #              object_name)
    #     dt0 = time.monotonic()
    #     try:
    #         _stage_demo()
    #     except Exception as e:  # noqa: BLE001
    #         log.warning("demo stage raised: %s — reporting success anyway",
    #                     e)
    #     # Mirror the real-grasp success path: hold the demo pose
    #     # 2 s for visual confirmation, then reset (close gripper
    #     # to 0.025 + park arm at init).
    #     log.info("DEMO MODE: holding demo pose 2.0s before post-pick reset")
    #     time.sleep(2.0)
    #     _safe_post_pick_reset("after demo success")
    #     return Pick_Response(
    #         success=True, message="ok (demo mode)",
    #         grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
    #         gripper_width=0.08, score=1.0,
    #         elapsed_s=time.monotonic() - dt0,
    #     )
    # ── END demo override ──

    # ── DEMO PLACE OVERRIDE ───────────────────────────────────────────
    # Sibling of the demo override block above — uncomment THIS one
    # instead (don't uncomment both; the first one to short-circuit
    # wins and the second is unreachable) when you want every
    # pick.pick(...) to play the "place" half of a pick-and-place
    # demo: drive to a fixed DEMO PLACE pose, hold 2 s, open gripper,
    # park at init.
    #
    # Same success semantics as the demo override: returns success=True
    # unconditionally, mirroring the real-grasp success path with a
    # 2 s pose-hold + post-pick reset (close gripper + park at init).
    # cpp side already opens the gripper as part of demo_place itself,
    # so the post-pick reset's controlGripper(0.025) immediately re-
    # closes it — that's intentional, mirrors the close-on-park
    # invariant of the real grasp pipeline.
    #
    # ── BEGIN demo_place override ──
    # if True:
    #     log.info("DEMO PLACE MODE: bypassing real grasp pipeline; "
    #              "calling manipulation/demo_place for object_name=%r",
    #              object_name)
    #     dt0 = time.monotonic()
    #     try:
    #         _stage_demo_place()
    #     except Exception as e:  # noqa: BLE001
    #         log.warning("demo_place stage raised: %s — "
    #                     "reporting success anyway", e)
    #     log.info("DEMO PLACE MODE: holding place pose 2.0s before "
    #              "post-pick reset")
    #     time.sleep(2.0)
    #     _safe_post_pick_reset("after demo_place success")
    #     return Pick_Response(
    #         success=True, message="ok (demo place mode)",
    #         grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
    #         gripper_width=0.08, score=1.0,
    #         elapsed_s=time.monotonic() - dt0,
    #     )
    # ── END demo_place override ──

    # ── VLA MODE ─────────────────────────────────────────────────────
    # In VLA mode, bypass the traditional detect→grasp→execute pipeline
    # entirely. Instead, call vla_client's execute MCP tool which runs
    # an end-to-end vision-language-action policy at 10Hz.
    if _mode == "vla":
        total_to = _DEFAULT_TIMEOUT_S
        t0 = time.monotonic()

        # Map object_name to a natural language instruction for VLA
        instruction = f"Pick up the {object_name}"
        log.info("VLA MODE: calling vla_execute with instruction=%r, "
                 "timeout=%.1fs, max_steps=%d",
                 instruction, total_to, _vla_max_steps)

        if "vla_execute" not in _endpoints:
            return Pick_Response(
                success=False,
                message="VLA mode: vla_execute endpoint not resolved on atlas",
                grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
                gripper_width=0.0, score=0.0, elapsed_s=0.0,
            )

        vla_resp = _mcp_call_sync(
            _endpoints["vla_execute"], "vla_execute", {
                "instruction": instruction,
                "timeout_s": total_to,
                "max_steps": _vla_max_steps,
            })

        elapsed = time.monotonic() - t0
        vla_success = vla_resp.get("success", False)
        vla_msg = vla_resp.get("message", "")
        vla_steps = int(vla_resp.get("steps_executed", 0))

        log.info("VLA MODE result: success=%s, steps=%d, msg=%r, elapsed=%.2fs",
                 vla_success, vla_steps, vla_msg[:60], elapsed)

        return Pick_Response(
            success=vla_success,
            message=f"vla: {vla_msg}" if vla_msg else "vla: done",
            grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
            gripper_width=0.0,
            score=1.0 if vla_success else 0.0,
            elapsed_s=elapsed,
        )
    # ── END VLA MODE ──────────────────────────────────────────────────

    total_to = _DEFAULT_TIMEOUT_S

    t0 = time.monotonic()
    deadline = t0 + total_to

    response: Pick_Response = Pick_Response(
        success=False, message="pick aborted before any stage ran",
        grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
        gripper_width=0.0, score=0.0, elapsed_s=0.0,
    )
    grasp_succeeded = False

    try:
        # Stage 0: park at the fixed observation pose before taking the
        # detector image. The 2D hand-eye homography is valid only from this
        # repeatable eye-in-hand camera pose.
        if "reset" in _endpoints:
            log.info("stage0 reset to fixed observation pose before detection")
            rst = _stage_reset()
            if not rst.get("success", False):
                msg = rst.get("_error") or rst.get("message", "unknown")
                response = Pick_Response(
                    success=False, message=f"pre_pick_reset_failed: {msg}",
                    grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
                    gripper_width=0.0, score=0.0,
                    elapsed_s=time.monotonic() - t0,
                )
                return response
        else:
            log.warning("stage0 reset skipped — manipulation/reset not resolved")

        # ── Stage 1: detect_object ─────────────────────────────────────────
        det = _stage_detect_object(object_name)
        if "_error" in det or not det.get("success"):
            msg = det.get("_error") or det.get("message", "unknown")
            response = Pick_Response(
                success=False, message=f"detection_failed: {msg}",
                grasp_pose=_build_pose_stamped_from_dict(_empty_pose_dict()),
                gripper_width=0.0, score=0.0,
                elapsed_s=time.monotonic() - t0,
            )
            return response

        bbox_2d   = list(det.get("bbox_2d") or [])
        center_3d = list(det.get("object_center_3d") or [])

        # ── Stage 2 + 3: grasp_request → execute_grasp, single attempt ──
        last_grasp_pose_dict: dict = _empty_pose_dict()
        last_gripper_width   = 0.0
        last_score           = 0.0
        last_failure_msg     = "no attempts made"

        if time.monotonic() >= deadline:
            response = Pick_Response(
                success=False,
                message=f"timeout: budget {total_to:.1f}s exhausted before grasp attempt",
                grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
                gripper_width=last_gripper_width, score=last_score,
                elapsed_s=time.monotonic() - t0,
            )
            return response

        # Stage 2: grasp_request. The downstream contract still accepts a
        # retry index; pass 0 because pick no longer retries.
        gr = _stage_grasp_request(object_name, bbox_2d, center_3d, 0)
        if "_error" in gr or not gr.get("success"):
            last_failure_msg = (
                f"grasp_pose failed: "
                f"{gr.get('_error') or gr.get('message', 'unknown')}")
            log.warning("%s", last_failure_msg)
            response = Pick_Response(
                success=False,
                message=f"grasp attempt failed; last: {last_failure_msg}",
                grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
                gripper_width=last_gripper_width, score=last_score,
                elapsed_s=time.monotonic() - t0,
            )
            return response

        last_grasp_pose_dict = gr.get("grasp_pose") or _empty_pose_dict()
        last_gripper_width   = float(gr.get("gripper_width", 0.0))
        last_score           = float(gr.get("score", 0.0))

        # Stage 3: three-stage vertical grasp motion.
        #
        # In vertical mode, grasp_request returns a pose at the
        # grasp height (z = z_table). We derive a pre_pose
        # (z + APPROACH_DIST) for the hover approach + lift.
        #
        # Motion sequence (mirrors roboarm.catch):
        #   3a. Move to pre_pose, gripper OPEN   (hover above)
        #   3b. Move to grasp_pose, gripper CLOSE (descend + grasp)
        #   3c. Move to pre_pose, gripper CLOSE   (lift)
        #
        # Use the remaining time budget so the whole pick respects the fixed
        # 60 s timeout. Each of the three execution stages gets a third.
        remaining = max(1.0, deadline - time.monotonic())
        stage_to = max(2.0, remaining / 3.0)

        pre_pose_dict = _lift_z(last_grasp_pose_dict, _APPROACH_DIST)

        # ── 3a: hover (pre_pose + gripper open) ──
        log.info("stage3a: approach to pre_pose (hover, gripper open)")
        eg_a = _stage_execute_grasp(pre_pose_dict, _GRIPPER_OPEN, stage_to)
        if "_error" in eg_a or not eg_a.get("success"):
            last_failure_msg = (
                f"execute failed (approach): "
                f"{eg_a.get('_error') or eg_a.get('message', 'unknown')}")
            log.warning("%s", last_failure_msg)
            response = Pick_Response(
                success=False,
                message=f"grasp attempt failed; last: {last_failure_msg}",
                grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
                gripper_width=last_gripper_width, score=last_score,
                elapsed_s=time.monotonic() - t0,
            )
            return response

        # ── 3b: descend + grasp (grasp_pose + gripper close) ──
        log.info("stage3b: descend to grasp_pose, close gripper")
        close_started_mono = time.monotonic()
        eg_b = _stage_execute_grasp(last_grasp_pose_dict, _GRIPPER_CLOSE,
                                    stage_to)
        if "_error" in eg_b or not eg_b.get("success"):
            last_failure_msg = (
                f"execute failed (descend): "
                f"{eg_b.get('_error') or eg_b.get('message', 'unknown')}")
            log.warning("%s", last_failure_msg)
            response = Pick_Response(
                success=False,
                message=f"grasp attempt failed; last: {last_failure_msg}",
                grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
                gripper_width=last_gripper_width, score=last_score,
                elapsed_s=time.monotonic() - t0,
            )
            return response

        # ── 3c: lift (pre_pose + gripper close) ──
        log.info("stage3c: lift to pre_pose, hold gripper close")
        eg_c = _stage_execute_grasp(pre_pose_dict, _GRIPPER_CLOSE, stage_to)
        if "_error" in eg_c or not eg_c.get("success"):
            # Lift failed — but we DID grasp the object. Log a warning but
            # still report success (the object is in the gripper, just not
            # lifted to hover height).
            log.warning("lift stage failed: %s — reporting success anyway "
                        "(object may be in gripper)",
                        eg_c.get("_error") or eg_c.get("message", ""))

        holding_ok, holding_msg, measured_gripper_width = (
            _verify_gripper_holding(close_started_mono)
        )
        if not holding_ok:
            last_failure_msg = f"gripper verification failed: {holding_msg}"
            log.warning("%s", last_failure_msg)
            response = Pick_Response(
                success=False,
                message=f"grasp attempt failed; last: {last_failure_msg}",
                grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
                gripper_width=last_gripper_width, score=last_score,
                elapsed_s=time.monotonic() - t0,
            )
            return response
        log.info("gripper verification ok: %s", holding_msg)

        grasp_succeeded = True
        response = Pick_Response(
            success=True,
            message=f"ok (vertical 3-stage grasp; {holding_msg})",
            grasp_pose=_build_pose_stamped_from_dict(last_grasp_pose_dict),
            gripper_width=measured_gripper_width, score=last_score,
            elapsed_s=time.monotonic() - t0,
        )
        return response
    finally:
        # Single chokepoint: every pick() exit path lands here.
        # On success: 10s pose-hold for visual confirmation, then reset.
        # On failure: reset immediately, no hold.
        # Either way, cpp ends up with clean state machine + arm at
        # init pose + gripper open, ready for the next pick.
        if grasp_succeeded:
            log.info("grasp successful — holding pose 10.0s before post-pick reset")
            time.sleep(10.0)
            _safe_post_pick_reset("after success")
        else:
            _safe_post_pick_reset("after failure")


# ── lifecycle ───────────────────────────────────────────────────────────────
@pick_skill.on_init
def init(cfg):
    """CMD_INIT: light. Don't query atlas — upstream services may
    still be warming up. cfg parsed for forward compat."""
    global _mode, _vla_max_steps
    global _gripper_open_width, _gripper_close_width
    global _gripper_grasp_threshold_width, _gripper_feedback_timeout_s
    global _gripper_joint_states_topic, _require_gripper_feedback
    global _GRIPPER_OPEN, _GRIPPER_CLOSE
    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")
    if "mode" in cfg:
        m = str(cfg["mode"]).strip().lower()
        if m in ("moveit", "vla"):
            _mode = m
        else:
            log.warning("ignoring invalid mode: %r (must be 'moveit' or 'vla')", cfg["mode"])
    if "vla_max_steps" in cfg:
        try:
            _vla_max_steps = int(cfg["vla_max_steps"])
        except (TypeError, ValueError):
            log.warning("ignoring invalid vla_max_steps: %r", cfg["vla_max_steps"])
    if "gripper_open_width" in cfg:
        try:
            _gripper_open_width = float(cfg["gripper_open_width"])
        except (TypeError, ValueError):
            log.warning("ignoring invalid gripper_open_width: %r",
                        cfg["gripper_open_width"])
    if "gripper_close_width" in cfg:
        try:
            _gripper_close_width = float(cfg["gripper_close_width"])
        except (TypeError, ValueError):
            log.warning("ignoring invalid gripper_close_width: %r",
                        cfg["gripper_close_width"])
    if "gripper_grasp_threshold_width" in cfg:
        try:
            _gripper_grasp_threshold_width = float(
                cfg["gripper_grasp_threshold_width"])
        except (TypeError, ValueError):
            log.warning("ignoring invalid gripper_grasp_threshold_width: %r",
                        cfg["gripper_grasp_threshold_width"])
    if "gripper_feedback_timeout_s" in cfg:
        try:
            _gripper_feedback_timeout_s = float(
                cfg["gripper_feedback_timeout_s"])
        except (TypeError, ValueError):
            log.warning("ignoring invalid gripper_feedback_timeout_s: %r",
                        cfg["gripper_feedback_timeout_s"])
    if "gripper_joint_states_topic" in cfg:
        _gripper_joint_states_topic = str(cfg["gripper_joint_states_topic"])
    if "require_gripper_feedback" in cfg:
        _require_gripper_feedback = bool(cfg["require_gripper_feedback"])

    _gripper_open_width = max(0.0, _gripper_open_width)
    _gripper_close_width = max(0.0, _gripper_close_width)
    _gripper_grasp_threshold_width = max(0.0, _gripper_grasp_threshold_width)
    _gripper_feedback_timeout_s = max(0.1, _gripper_feedback_timeout_s)
    _GRIPPER_OPEN = _gripper_open_width
    _GRIPPER_CLOSE = _gripper_close_width

    log.info(
        "CMD_INIT ok (mode=%s, timeout_s=%.1f, "
        "grasp_attempts=1, vla_max_steps=%d, gripper_open=%.3f, "
        "gripper_close=%.3f, grasp_threshold=%.4f, feedback_topic=%s)",
        _mode, _DEFAULT_TIMEOUT_S, _vla_max_steps,
        _GRIPPER_OPEN, _GRIPPER_CLOSE, _gripper_grasp_threshold_width,
        _gripper_joint_states_topic,
    )
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
        if _mode == "moveit":
            _start_gripper_monitor()
    log.info("CMD_ACTIVATE ok — endpoints resolved: %s", list(_endpoints.keys()))
    return Ok()


@pick_skill.on_deactivate
def deactivate():
    """CMD_DEACTIVATE: drop client cache + endpoints. Idle eviction."""
    global _endpoints
    with _state_lock, _mcp_clients_lock:
        _endpoints = None
        _mcp_clients.clear()
        _stop_gripper_monitor()
    log.info("CMD_DEACTIVATE ok — endpoints cleared")
    return Ok()


def main() -> int:
    pick_skill.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
