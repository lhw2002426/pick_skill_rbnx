# pick_skill_rbnx

User-facing `pick(object_name)` skill for the Piper + Orbbec vertical grasp
pipeline.

## What It Does

Owns `robonix/skill/pick/*` and exposes two MCP tools:

```text
robonix/skill/pick/pick
robonix/skill/pick/put_down
```

`pick` accepts only:

```json
{"object_name": "comb"}
```

It uses a fixed 60 s budget and makes one grasp attempt.

`put_down` accepts an empty request and releases the currently held object by
opening the gripper to maximum width before returning the arm to the teach-safe
pose.

## Runtime Path

```text
pilot / executor
  -> pick_skill.pick(object_name)
  -> llm_detect.detect_object
  -> grasp_pose.grasp_request
  -> roboarm_ik.execute_grasp
  -> return while holding the object

pilot / executor
  -> pick_skill.put_down()
  -> roboarm_ik.teach_safe
```

Upstream calls are resolved through Atlas. `detect_object` currently uses MCP
HTTP; `grasp_request`, `execute_grasp`, and `reset` use gRPC. The skill does
not call legacy ROS services or `/graspnet` topics.

## Lifecycle

```text
rbnx boot:
  CMD_INIT only, skill remains INITIALIZED

first pick call:
  CMD_ACTIVATE resolves llm_detect + grasp_pose + roboarm_ik endpoints

idle/deactivate:
  endpoint and client caches are cleared
```

## Pipeline

1. Reset/park to the fixed observation pose when `manipulation/reset` is
   available.
2. Detect `object_name` with `llm_detect`.
3. Compute a vertical grasp pose with `grasp_pose`.
4. Execute the three-stage motion through `roboarm_ik`:
   approach above target with gripper open, descend/close, lift.
5. Verify gripper feedback on `/arm/joint_states_single`.
6. On success, leave the object held until `put_down` is called.
7. On failure, reset/park immediately for recovery.

## Config

The default deploy needs no `pick` config. Gripper feedback is required:
if no fresh `/arm/joint_states_single` gripper width arrives after the close
command, the pick fails.

## Build

```bash
bash scripts/build.sh
```

Build is codegen-only; there is no vendored ROS package.
