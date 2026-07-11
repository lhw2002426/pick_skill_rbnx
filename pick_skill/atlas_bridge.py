# SPDX-License-Identifier: MulanPSL-2.0
"""pick_skill_rbnx atlas bridge — Capability + contract-typed MCP tool.

Orchestrates the three upstream perception+manipulation services into
a single user-invocable `pick(object_name)` operation, exposed as the
`robonix/skill/pick/pick` MCP tool.

Pipeline (atlas-resolved endpoints, no hardcoded ROS topic / service names):

    Pilot LLM  ──pick("the comb")──►  this skill
                                          │
                  ┌────────── 1. detect_object ─────────────►  llm_detect_rbnx
                  │            (object_name → bbox + 3D center)
                  ▼
                  ┌──── 2. grasp_request ────►  grasp_pose_rbnx
                  │       (name + bbox + center + retry=0 → grasp_pose)
                      │
                      ▼
                      ┌──── 3. execute_grasp ────►  roboarm_ik_rbnx
                      │       (pre_pose + gripper full-open,
                      │        grasp_pose + gripper closed,
                      │        lift + gripper closed)
                      │
                      ▼
                  success → verify gripper did not close past threshold
                                → return with object held in the gripper
                                → return success
                  On timeout, execution failure, or detection failure:
                  call reset before returning, so the next pick starts
                  from a known arm state.

    Pilot LLM  ──put_down()──────►  this skill
                                          │
                                          └──── teach_safe ─► roboarm_ik_rbnx
                                                (open gripper + teach-safe pose)

Why robonix contracts instead of raw ROS service calls (the upstream pick.py
path):

* Consistent contract surface — pilot's view of the system is the
  set of robonix/* contracts; mixing ROS services means the LLM sees
  inconsistent capability shapes for "morally equivalent" calls.
* Topology independence — if a future deploy reshuffles which
  package owns which service, atlas resolution adapts; hardcoded
  topic names don't.
* The manipulation provider owns motion execution and its own timeout.
  We subscribe to /arm/joint_states_single to verify the gripper did
  not close completely after the grasp,
  mirroring roboarm's "closed too far means empty" success check.

Lifecycle (Skill — lazy activate, same as explore_rbnx):
    on_init      — light. State machine reaches INITIALIZED at boot.
                   No atlas resolution here (upstream services may
                   still be warming up).
    on_activate  — heavy. Sent by executor on first MCP call into
                   any of this skill's tools. Resolve the three
                   upstream endpoints, build clients lazily,
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
# Resolved upstream endpoints. None until on_activate completes.
# Three keys, three contract ids — see REQUIRED_INPUTS below.
_endpoints: Optional[dict[str, str]] = None
_DEFAULT_TIMEOUT_S   = 60.0
_vla_max_steps       = 50
_gripper_open_width  = 0.08
_gripper_close_width = 0.0
_gripper_grasp_threshold_width = 0.005
_gripper_feedback_timeout_s = 2.0
_gripper_joint_states_topic = "/arm/joint_states_single"

# Execution mode: "pipeline" (detect→grasp→execute) or "vla".
_mode = "pipeline"

# We deliberately keep ONE FastMCP Client per upstream URL (lazily
# constructed in the handler, not on_activate). FastMCP's Client is
# async-context-manager based: each call_tool happens inside an
# `async with client as c: ...` block. We don't want to build/tear
# the client per-call (3 round-trips per pick = 3× connect/handshake
# overhead), so we cache the URL→Client mapping and reuse.
_mcp_clients_lock = threading.Lock()
_mcp_clients: dict[str, Any] = {}   # base_url → fastmcp.Client
_grpc_clients_lock = threading.Lock()
_grpc_channels: dict[str, Any] = {}  # host:port → grpc.Channel
_grpc_stubs: dict[str, Any] = {}     # "kind@endpoint" → generated stub

_gripper_state_lock = threading.Lock()
_gripper_latest_width: Optional[float] = None
_gripper_latest_seen_mono: Optional[float] = None
_gripper_monitor_node: Any = None
_gripper_monitor_thread: Optional[threading.Thread] = None
_gripper_monitor_stop = threading.Event()


# ── atlas-resolved upstream contracts ───────────────────────────────────────
REQUIRED_INPUTS = {
    "detect_object":  ("robonix/service/perception/object_detect/detect_object", "mcp"),
    "grasp_request":  ("robonix/service/perception/grasp_pose/grasp_request",    "grpc"),
    "execute_grasp":  ("robonix/service/manipulation/execute_grasp",             "grpc"),
}

# Optional upstreams — pick still activates if these are missing on
# atlas, but each key has degraded behaviour documented:
#   "reset" — pre-pick observation pose, failure recovery, and explicit
#             failure recovery.
#   "teach_safe" — explicit put_down implementation. If unavailable,
#                  successful pick still leaves the object held, but put_down
#                  cannot run.
OPTIONAL_INPUTS = {
    "reset":          ("robonix/service/manipulation/reset",                     "grpc"),
    "teach_safe":     ("robonix/service/manipulation/teach_safe",                "grpc"),
    "vla_execute":    ("robonix/skill/vla/execute",                              "mcp"),
}

# In VLA mode, vla_execute becomes required instead of the traditional pipeline
VLA_REQUIRED_INPUTS = {
    "vla_execute":    ("robonix/skill/vla/execute",                              "mcp"),
}


def _resolve_inputs(deadline_s: float = 60.0) -> dict[str, str]:
    """Block until atlas can resolve all REQUIRED_INPUTS upstream endpoints,
    or fail loudly. Then best-effort resolve OPTIONAL_INPUTS
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
                f"llm_detect_rbnx (object_detect) + grasp_pose_rbnx (grasp_pose) "
                f"+ roboarm_ik_rbnx (manipulation/execute_grasp) all ACTIVE "
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


# ── gRPC client helpers ─────────────────────────────────────────────────────
def _grpc_channel_for(endpoint: str):
    """Return a cached grpc.Channel for an atlas-resolved host:port."""
    with _grpc_clients_lock:
        ch = _grpc_channels.get(endpoint)
        if ch is not None:
            return ch
        import grpc
        ch = grpc.insecure_channel(
            endpoint, options=[("grpc.enable_http_proxy", 0)])
        _grpc_channels[endpoint] = ch
        return ch


def _grasp_request_stub_for(endpoint: str):
    key = f"grasp_request@{endpoint}"
    with _grpc_clients_lock:
        stub = _grpc_stubs.get(key)
        if stub is not None:
            return stub
    import robonix_contracts_pb2_grpc as contracts_grpc

    stub = contracts_grpc.RobonixServicePerceptionGraspPoseGraspRequestStub(
        _grpc_channel_for(endpoint))
    with _grpc_clients_lock:
        _grpc_stubs[key] = stub
    return stub


def _execute_grasp_stub_for(endpoint: str):
    key = f"execute_grasp@{endpoint}"
    with _grpc_clients_lock:
        stub = _grpc_stubs.get(key)
        if stub is not None:
            return stub
    import robonix_contracts_pb2_grpc as contracts_grpc

    stub = contracts_grpc.RobonixServiceManipulationExecuteGraspStub(
        _grpc_channel_for(endpoint))
    with _grpc_clients_lock:
        _grpc_stubs[key] = stub
    return stub


def _reset_stub_for(endpoint: str):
    key = f"reset@{endpoint}"
    with _grpc_clients_lock:
        stub = _grpc_stubs.get(key)
        if stub is not None:
            return stub
    import robonix_contracts_pb2_grpc as contracts_grpc

    stub = contracts_grpc.RobonixServiceManipulationResetStub(
        _grpc_channel_for(endpoint))
    with _grpc_clients_lock:
        _grpc_stubs[key] = stub
    return stub


def _teach_safe_stub_for(endpoint: str):
    key = f"teach_safe@{endpoint}"
    with _grpc_clients_lock:
        stub = _grpc_stubs.get(key)
        if stub is not None:
            return stub
    import robonix_contracts_pb2_grpc as contracts_grpc

    stub = contracts_grpc.RobonixServiceManipulationTeachSafeStub(
        _grpc_channel_for(endpoint))
    with _grpc_clients_lock:
        _grpc_stubs[key] = stub
    return stub


def _pose_stamped_pb_to_dict(msg: Any) -> dict:
    return {
        "header": {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "frame_id": str(msg.header.frame_id),
        },
        "pose": {
            "position": {
                "x": float(msg.pose.position.x),
                "y": float(msg.pose.position.y),
                "z": float(msg.pose.position.z),
            },
            "orientation": {
                "x": float(msg.pose.orientation.x),
                "y": float(msg.pose.orientation.y),
                "z": float(msg.pose.orientation.z),
                "w": float(msg.pose.orientation.w),
            },
        },
    }


def _pose_stamped_dict_to_pb(d: dict) -> Any:
    import builtin_interfaces_pb2
    import geometry_msgs_pb2
    import std_msgs_pb2

    header = d.get("header", {}) or {}
    stamp = header.get("stamp", {}) or {}
    pose = d.get("pose", {}) or {}
    pos = pose.get("position", {}) or {}
    ori = pose.get("orientation", {}) or {}
    return geometry_msgs_pb2.PoseStamped(
        header=std_msgs_pb2.Header(
            stamp=builtin_interfaces_pb2.Time(
                sec=int(stamp.get("sec", 0)),
                nanosec=int(stamp.get("nanosec", 0)),
            ),
            frame_id=str(header.get("frame_id", "arm/base_link")),
        ),
        pose=geometry_msgs_pb2.Pose(
            position=geometry_msgs_pb2.Point(
                x=float(pos.get("x", 0.0)),
                y=float(pos.get("y", 0.0)),
                z=float(pos.get("z", 0.0)),
            ),
            orientation=geometry_msgs_pb2.Quaternion(
                x=float(ori.get("x", 0.0)),
                y=float(ori.get("y", 0.0)),
                z=float(ori.get("z", 0.0)),
                w=float(ori.get("w", 1.0)),
            ),
        ),
    )


def _grasp_request_grpc_sync(endpoint: str, args: dict) -> dict:
    """Call grasp_pose.grasp_request over gRPC and return MCP-shape dict."""
    try:
        import grasp_pb2

        req = grasp_pb2.GraspRequest_Request(
            object_name=str(args.get("object_name", "")),
            retry=int(args.get("retry", 0)),
        )
        req.bbox_2d.extend(float(x) for x in args.get("bbox_2d", []) or [])
        req.object_center_3d.extend(
            float(x) for x in args.get("object_center_3d", []) or [])
        resp = _grasp_request_stub_for(endpoint).GraspRequest(req)
        return {
            "grasp_pose": _pose_stamped_pb_to_dict(resp.grasp_pose),
            "gripper_width": float(resp.gripper_width),
            "score": float(resp.score),
            "success": bool(resp.success),
            "message": str(resp.message),
        }
    except Exception as e:  # noqa: BLE001
        log.warning("grpc call grasp_request failed: %s", e)
        return {"_error": str(e)}


def _execute_grasp_grpc_sync(endpoint: str, args: dict) -> dict:
    """Call roboarm_ik.execute_grasp over gRPC and return dict response."""
    try:
        import manipulation_pb2

        req = manipulation_pb2.ExecuteGrasp_Request(
            target_pose=_pose_stamped_dict_to_pb(args.get("target_pose", {}) or {}),
            gripper_width=float(args.get("gripper_width", 0.0)),
            timeout_s=float(args.get("timeout_s", 0.0)),
        )
        resp = _execute_grasp_stub_for(endpoint).ExecuteGrasp(req)
        return {
            "success": bool(resp.success),
            "message": str(resp.message),
            "elapsed_s": float(resp.elapsed_s),
        }
    except Exception as e:  # noqa: BLE001
        log.warning("grpc call execute_grasp failed: %s", e)
        return {"_error": str(e)}


def _reset_grpc_sync(endpoint: str) -> dict:
    """Call roboarm_ik.reset over gRPC and return dict response."""
    try:
        import manipulation_pb2

        resp = _reset_stub_for(endpoint).Reset(
            manipulation_pb2.Reset_Request(ack=True))
        return {
            "success": bool(resp.success),
            "message": str(resp.message),
            "elapsed_s": float(resp.elapsed_s),
        }
    except Exception as e:  # noqa: BLE001
        log.warning("grpc call reset failed: %s", e)
        return {"_error": str(e)}


def _teach_safe_grpc_sync(endpoint: str) -> dict:
    """Call roboarm_ik.teach_safe over gRPC and return dict response."""
    try:
        import manipulation_pb2

        resp = _teach_safe_stub_for(endpoint).TeachSafe(
            manipulation_pb2.TeachSafe_Request(ack=True))
        return {
            "success": bool(resp.success),
            "message": str(resp.message),
            "elapsed_s": float(resp.elapsed_s),
        }
    except Exception as e:  # noqa: BLE001
        log.warning("grpc call teach_safe failed: %s", e)
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

    The gripper success signal is only available as ROS JointState feedback
    today.
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
        return False, msg, 0.0
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
    """Park through the active manipulation provider and open the gripper."""
    assert _endpoints is not None
    if "reset" not in _endpoints:
        log.warning("reset SKIPPED — "
                    "manipulation/reset not on atlas")
        return {"success": True, "message": "skipped (capability missing)",
                "elapsed_s": 0.0}
    log.info("reset (open gripper, park arm at init)")
    resp = _reset_grpc_sync(_endpoints["reset"])
    log.info("reset result: success=%s msg=%r elapsed=%.2fs",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("elapsed_s", 0.0)))
    return resp


def _stage_teach_safe() -> dict:
    """Open the gripper and park at the configured teach-safe pose."""
    assert _endpoints is not None
    if "teach_safe" not in _endpoints:
        log.warning("teach_safe SKIPPED — "
                    "manipulation/teach_safe not on atlas")
        return {"success": False, "message": "teach_safe capability missing",
                "elapsed_s": 0.0}
    log.info("teach_safe (open gripper, park arm at teach-safe pose)")
    resp = _teach_safe_grpc_sync(_endpoints["teach_safe"])
    log.info("teach_safe result: success=%s msg=%r elapsed=%.2fs",
             resp.get("success"), resp.get("message", "")[:60],
             float(resp.get("elapsed_s", 0.0)))
    return resp


def _safe_post_pick_reset(context: str) -> None:
    """Best-effort post-pick reset, never raises.

    Used by every pick() exit path (success AND failure) to leave
    the arm in a clean state for the next pick. Logs but swallows any
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
    resp = _grasp_request_grpc_sync(_endpoints["grasp_request"], args)
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
    resp = _execute_grasp_grpc_sync(_endpoints["execute_grasp"], args)
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
# match grasp_pose_rbnx's config/vertical_grasp.yaml approach_dist.
_APPROACH_DIST = 0.10   # m

# Gripper widths for the three stages.
_GRIPPER_OPEN  = _gripper_open_width
_GRIPPER_CLOSE = _gripper_close_width


# ── MCP tool (typed against codegen Pick_Request/Pick_Response) ─────────────
from pick_mcp import (  # noqa: E402  pylint: disable=wrong-import-position
    Pick_Request, Pick_Response, PutDown_Request, PutDown_Response,
)
# Nested types we need to instantiate when building the response.
from geometry_msgs_mcp import (  # noqa: E402
    PoseStamped, Pose, Point, Quaternion,
)
from std_msgs_mcp import Header  # noqa: E402
from builtin_interfaces_mcp import Time  # noqa: E402


def _build_pose_stamped_from_dict(d: dict) -> PoseStamped:
    """Build the codegen PoseStamped dataclass from an MCP-shape dict.
    The dict structure must match the PoseStamped to_dict() shape returned by
    grasp_pose and the manipulation provider."""
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
    with llm_detect, plans a grasp pose with the geometric grasp_pose
    estimator, then executes via roboarm_ik + the Piper arm.

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
        if grasp_succeeded:
            log.info("grasp successful — leaving object held until put_down is called")
        else:
            _safe_post_pick_reset("after failure")


@pick_skill.mcp("robonix/skill/pick/put_down")
def put_down(_req: PutDown_Request) -> PutDown_Response:
    """Put down the currently held object.

    This tool is intentionally exposed as an MCP skill tool so the pilot LLM
    can call it directly after a successful pick. It releases the gripper and
    returns the arm to the configured teach-safe pose by delegating to the
    active manipulation/teach_safe provider.
    """
    if _endpoints is None:
        return PutDown_Response(
            success=False,
            message="pick skill not active (atlas hasn't resolved upstream services yet)",
            elapsed_s=0.0,
        )
    if "teach_safe" not in _endpoints:
        return PutDown_Response(
            success=False,
            message="put_down failed: manipulation/teach_safe endpoint not resolved on atlas",
            elapsed_s=0.0,
        )

    t0 = time.monotonic()
    resp = _stage_teach_safe()
    if "_error" in resp:
        return PutDown_Response(
            success=False,
            message=f"put_down failed: {resp['_error']}",
            elapsed_s=time.monotonic() - t0,
        )
    return PutDown_Response(
        success=bool(resp.get("success", False)),
        message=str(resp.get("message", "put_down complete")),
        elapsed_s=float(resp.get("elapsed_s", time.monotonic() - t0)),
    )


# ── lifecycle ───────────────────────────────────────────────────────────────
@pick_skill.on_init
def init(cfg):
    """CMD_INIT: light. Don't query atlas — upstream services may
    still be warming up. cfg parsed for forward compat."""
    global _mode, _vla_max_steps
    global _gripper_open_width, _gripper_close_width
    global _gripper_grasp_threshold_width, _gripper_feedback_timeout_s
    global _gripper_joint_states_topic
    global _GRIPPER_OPEN, _GRIPPER_CLOSE
    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")
    if "mode" in cfg:
        m = str(cfg["mode"]).strip().lower()
        if m in ("pipeline", "vla"):
            _mode = m
        else:
            log.warning("ignoring invalid mode: %r (must be 'pipeline' or 'vla')", cfg["mode"])
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
    """CMD_ACTIVATE: heavy. Resolve upstream endpoints. Idempotent."""
    global _endpoints
    with _state_lock:
        if _endpoints is not None:
            log.info("CMD_ACTIVATE — already active, no-op")
            return Ok()
        try:
            _endpoints = _resolve_inputs()
        except RuntimeError as e:
            return Err(str(e))
        if _mode == "pipeline":
            _start_gripper_monitor()
    log.info("CMD_ACTIVATE ok — endpoints resolved: %s", list(_endpoints.keys()))
    return Ok()


@pick_skill.on_deactivate
def deactivate():
    """CMD_DEACTIVATE: drop client cache + endpoints. Idle eviction."""
    global _endpoints
    with _state_lock, _mcp_clients_lock, _grpc_clients_lock:
        _endpoints = None
        _mcp_clients.clear()
        for ch in _grpc_channels.values():
            try:
                ch.close()
            except Exception:  # noqa: BLE001
                pass
        _grpc_channels.clear()
        _grpc_stubs.clear()
        _stop_gripper_monitor()
    log.info("CMD_DEACTIVATE ok — endpoints cleared")
    return Ok()


def main() -> int:
    pick_skill.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
