#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
#
# Build phase: rbnx codegen --mcp ONLY.
#
# Skill is pure Python with NO vendored ROS packages — it talks to
# upstream services exclusively over MCP HTTP via FastMCP's Client.
# So no colcon build, no graspnet_msgs, no piper_msgs in src/. The
# only build-time concern is generating:
#   * atlas_pb2 / atlas_pb2_grpc                 (Skill.run() runtime)
#   * pick_mcp.py                                 (Pick_Request/_Response)
#   * geometry_msgs_mcp.py / std_msgs_mcp.py /
#     builtin_interfaces_mcp.py                   (PoseStamped + nested,
#                                                  needed because Pick.srv
#                                                  carries a PoseStamped)
#
# Soft-warn if FastMCP isn't installed (pip install fastmcp). It's a
# runtime requirement, not build-time, so we don't fail.
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"
CLEAN="${RBNX_BUILD_CLEAN:-}"

if [[ "$CLEAN" == "1" ]]; then
    echo "[pick_skill/build] clean: removing rbnx-build/"
    rm -rf rbnx-build
fi
mkdir -p rbnx-build/data

# Sanity: warn if FastMCP isn't pip-installed yet.
if ! python3 -c "import fastmcp" 2>/dev/null; then
    echo "[pick_skill/build] NOTE: fastmcp not importable. Install with:"
    echo "                       pip install fastmcp"
    echo "                     (deploy will fail at first MCP call without it)."
fi

FLAGS=(--out-dir "$PKG/rbnx-build/codegen" --mcp)
[[ "$CLEAN" == "1" ]] && FLAGS+=(--clean)
echo "[pick_skill/build] rbnx codegen ${FLAGS[*]}"
rbnx codegen -p "$PKG" "${FLAGS[@]}"

touch "$PKG/rbnx-build/.rbnx-built"
echo "[pick_skill/build] done."
