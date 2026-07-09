# pick_skill_rbnx

Robonix package for the user-facing "pick up X" skill on the Piper +
Orbbec Dabai DCW grasp pipeline. Stage 6 of the migration. The
LLM-facing entry point of the whole pipeline.

## What it does

Owns `robonix/skill/pick/*`. Exposes a single MCP tool —
`robonix/skill/pick/pick(object_name)` — that orchestrates the three
upstream services into one synchronous "pick the X" call:

```
                 Pilot LLM
                     │
             pick("the comb")          (MCP)
                     │
                     ▼
            ┌────────────────────────┐
            │   pick_skill_rbnx      │
            │ (Skill, this package)  │
            └────────┬───────────────┘
                     │
        ┌────────────┼─────────────┐
        │            │             │
        ▼            ▼             ▼
   detect_object  grasp_request  execute_grasp
   (yolo_world)  (yolo_grasp)   (piper_moveit)
        │            │             │
        ▼            ▼             ▼
       2D bbox    grasp pose     arm motion
       + 3D ctr   + width        + gripper close
```

All three upstream calls are **MCP HTTP** (FastMCP Client). atlas
resolves the URLs at on_activate time. There is **no fallback** to
the legacy ROS service path (`/yolo/detect_object`,
`/graspnet/grasp_request`) — Stage 6's whole point is the
LLM-facing surface speaks robonix contracts only.

## What this means for the legacy ROS surfaces

Stages 4A/4B/5 deliberately kept their **ROS service** surfaces alive
alongside the new MCP ones (so legacy `pick.py` could keep running
during migration). With Stage 6 in place, **the LLM never touches
those ROS surfaces** — it goes through this skill, which goes through
MCP only.

The legacy ROS surfaces themselves are still there (other consumers
in the live system might still depend on them). They'll be removed
in a separate, post-Stage-6 cleanup pass when nothing else needs them.

## Lifecycle (lazy activate)

Same shape as `explore_rbnx`. Skills are different from primitives /
services here — they don't auto-bring-up on `rbnx boot`:

```
boot:
  CMD_INIT  ── light: parse cfg, log defaults. State = INITIALIZED.
              (No atlas resolution — upstream services may still be
               warming up, and we don't want to deadlock boot order.)

first MCP call:
  CMD_ACTIVATE ── heavy: block up to 60s waiting for atlas to
                  resolve the three upstream MCP endpoints. State = ACTIVE.

idle:
  CMD_DEACTIVATE ── drop endpoint cache + FastMCP client cache.
                    Re-ACTIVATE rebuilds. (Eviction policy is
                    executor's, not ours.)
```

## Architecture

```
pick_skill_rbnx/
├── package_manifest.yaml
├── capabilities/
│   ├── driver.v1.toml             # rpc, lifecycle/srv/Driver.srv
│   ├── pick.v1.toml               # rpc/MCP, pick/srv/Pick.srv
│   └── lib/pick/srv/
│       └── Pick.srv               # codegen → Pick_Request/_Response
├── pick_skill/
│   ├── __init__.py
│   └── atlas_bridge.py            # Skill + on_activate + MCP orchestration
├── scripts/
│   ├── build.sh                   # rbnx codegen --mcp ONLY (no colcon!)
│   └── start.sh                   # source ROS, exec atlas_bridge
└── (no src/ — pure Python, no vendored ROS packages)
```

`src/` is intentionally absent. Unlike Stages 1–5 we don't vendor
any ROS source: the skill talks to upstream services exclusively
over MCP HTTP, and our own surfaces (Pick.srv codegen) only need
Python dataclasses. Build is significantly faster (~2s vs ~30–60s
for piper_moveit_rbnx).

## Pipeline implementation

`pick_skill/atlas_bridge.py::pick(req)` runs:

1. **Stage 1** — `detect_object(req.object_name)` over MCP. If
   `success=False` (object not in view, low confidence, etc.):
   short-circuit return `detection_failed: <message>`.

2. **Stage 2** — `grasp_request(name, bbox_2d, object_center_3d, retry=0)`.
   The downstream grasp service still accepts a retry index; pick now
   always passes `0` and makes exactly one grasp attempt.

3. **Stage 3** — `execute_grasp(grasp_pose, gripper_width, exec_timeout)`.
   The execute_grasp service itself does the busy/idle wait on
   `/arm/arm_status`, so we don't need to. Returns once the arm
   hit busy → idle (success) or budget exhausted (timeout).

4. If the single attempt fails, return
   `grasp attempt failed; last: <message>`.

## Time budgeting

The total pick budget is fixed at 60 s. Internally distributed:

* Detection: bounded by FastMCP client timeout (~5–10s typical).
* Grasp request: also FastMCP-bounded.
* Execute grasp: the remaining budget is split across the three
  vertical-grasp execution stages.

## Config

```yaml
skill:
  - name: pick
    config:
      gripper_open_width: 0.08
```

The `pick` MCP tool accepts only `object_name`; timeout and retry
count are not caller-configurable.

## Build / run

```bash
# Standalone
cd /Users/howenliu/lab/packages/pick_skill_rbnx
bash scripts/build.sh

# Integrated
cd /Users/howenliu/lab/piper_grasp_deploy
rbnx boot
```

## Verification (in order)

```bash
# 1. atlas-side: provider visible, INITIALIZED at boot
rbnx caps | grep pick
# expect: pick  com.robonix.skill.pick  INITIALIZED
#   robonix/skill/pick/driver  (rpc/grpc)
#   robonix/skill/pick/pick    (rpc/mcp)

# 2. End-to-end via Pilot LLM (the design intent):
rbnx ask "请帮我把那个 comb 抓起来"
# pilot should:
#   * pick the `pick` MCP tool
#   * call it with object_name="comb"
#   * return the grasp result to the user
# Behind the scenes pick_skill triggers CMD_ACTIVATE (resolves
# yolo_world + yolo_grasp + piper_moveit MCP endpoints), then runs
# the 3-stage pipeline.

# 3. Direct MCP test (skip pilot):
curl -s http://127.0.0.1:<pick_skill_port>/mcp/ \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc":"2.0", "id":1,
      "method":"tools/call",
      "params":{
        "name":"pick",
        "arguments":{"object_name":"comb"}
      }
    }'
# Find the port via `rbnx caps -v` or the package's start log.
```

## Failure modes

| symptom | cause | fix |
|---|---|---|
| `pick skill cannot find dependencies on atlas: missing [...]` | one of yolo_world / yolo_grasp / piper_moveit not ACTIVE | `rbnx caps` to see which one's missing; check that package's log |
| `detection_failed: object 'X' not found at confidence ≥ 0.2` | YOLOE genuinely doesn't see the object | move closer; try a different object_name; check `/yolo/detect_object` independently |
| `grasp_pose failed: PLACEHOLDER` | yolo_grasp_rbnx still in PLACEHOLDER mode (real estimator not wired) | see yolo_grasp_rbnx README "Cutover steps" |
| `execute failed (approach): timeout: arm_status never went busy` | manipulation provider didn't pick up the GraspPose, OR /arm/arm_status not flowing | check piper_ctl_rbnx ACTIVE + manipulation provider logs |
| `pick skill not active` on first call | CMD_ACTIVATE failed — atlas_bridge logs the actual reason | check pick_skill log; usually means upstream service down |
| Hangs for full timeout | one of the upstream MCP servers is reachable but not responding | check the slow service's log; the pick budget is fixed at 60 s |

## Coupling with neighbors

* **Upstream** yolo_world_rbnx (Stage 4A) — provides
  `service/perception/object_detect/detect_object` MCP.
* **Upstream** yolo_grasp_rbnx (Stage 4B) — provides
  `service/perception/grasp_pose/grasp_request` MCP. **Currently in
  PLACEHOLDER mode** until real estimator is wired (see that
  package's README) — pick will return failure until then.
* **Upstream** piper_moveit_rbnx (Stage 5) — provides
  `service/manipulation/execute_grasp` MCP.
* **Indirectly upstream** piper_ctl + piper_description +
  easy_handeye2 — pick doesn't talk to them, but execute_grasp does
  (transitively); if they're not ACTIVE, execute_grasp fails fast
  and pick reports it.

## Why not vendor pick.py?

Upstream `pick.py` uses raw rclpy: ROS service clients +
`/arm_status` topic subscription. Stage 6's purpose is to replace
that path with atlas-routed MCP, which means starting from the same
algorithmic skeleton (detect → grasp → wait for arm) but rewriting
the I/O layer entirely. Vendoring pick.py would just be carrying
dead code (the rclpy parts), so we don't — we re-derive the orchestration
in `atlas_bridge.py` against the new typed contracts.

The ~50 lines of orchestration logic in `pick(req)` are a faithful
translation of pick.py's main loop, minus the ROS-specific I/O.
