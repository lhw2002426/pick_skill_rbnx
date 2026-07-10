# pick_skill_rbnx

User-facing `pick(object_name)` skill for the Piper + Orbbec vertical grasp
pipeline.

## What It Does

Owns `robonix/skill/pick/*` and exposes one MCP tool:

```text
robonix/skill/pick/pick
```

The tool accepts only:

```json
{"object_name": "comb"}
```

It uses a fixed 60 s budget and makes one grasp attempt.

## Runtime Path

```text
pilot / executor
  -> pick_skill.pick(object_name)
  -> llm_detect.detect_object
  -> grasp_pose.grasp_request
  -> roboarm_ik.execute_grasp
  -> roboarm_ik.reset
```

All upstream calls are MCP HTTP calls resolved through Atlas. The skill does
not call legacy ROS services or `/graspnet` topics.

## Lifecycle

```text
rbnx boot:
  CMD_INIT only, skill remains INITIALIZED

first pick call:
  CMD_ACTIVATE resolves llm_detect + grasp_pose + roboarm_ik endpoints

idle/deactivate:
  endpoint and FastMCP client caches are cleared
```

## Pipeline

1. Reset/park to the fixed observation pose when `manipulation/reset` is
   available.
2. Detect `object_name` with `llm_detect`.
3. Compute a vertical grasp pose with `grasp_pose`.
4. Execute the three-stage motion through `roboarm_ik`:
   approach above target with gripper open, descend/close, lift.
5. Verify gripper feedback on `/arm/joint_states_single`.
6. Hold briefly on success, then reset/park for the next pick.

## Config

The default deploy needs no `pick` config. Gripper feedback is required:
if no fresh `/arm/joint_states_single` gripper width arrives after the close
command, the pick fails.

## Build

```bash
bash scripts/build.sh
```

Build is codegen-only; there is no vendored ROS package.
