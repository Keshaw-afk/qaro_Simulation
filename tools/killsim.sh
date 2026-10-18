#!/usr/bin/env bash
# Stop every running Gazebo simulation. This lives in a file so the pkill pattern can never match
# the command line of whoever calls it (pkill -f on an inline command kills its own shell).
pkill -f '[g]z sim'
sleep 2
echo "sims left: $(pgrep -fc '[g]z sim')"
