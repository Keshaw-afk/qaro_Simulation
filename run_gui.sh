#!/usr/bin/env bash
# Start the Swarm Nav RL desktop app (run inside the distrobox container).
cd "$(dirname "$0")" && exec python3 swarm_gui.py
