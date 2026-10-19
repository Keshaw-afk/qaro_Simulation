# Convenience targets. Run `make help`. Gazebo targets need ROS sourced by the recipe (ROS_SETUP) and a working Gazebo Harmonic.
SHELL := bash
ROS_SETUP ?= /opt/ros/jazzy/setup.bash
VENV_PY := .venv/bin/python
WEIGHTS ?= weights/final_policy.npz
FIRST ?= 1
N ?= 3

.PHONY: help check-env test gui eval-small eval-big eval-2d train

help:            ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

check-env:       ## verify system dependencies
	source $(ROS_SETUP) && python3 tools/check_env.py

test:            ## fast tests, no simulator (parity test only if .venv exists)
	python3 tests/test_layout.py
	python3 tests/test_shield.py
	@if [ -x $(VENV_PY) ]; then $(VENV_PY) tests/test_obs_parity.py; else echo "skipping parity test: no .venv (see docs/SETUP.md)"; fi

gui:             ## desktop app
	source $(ROS_SETUP) && python3 swarm_gui.py

eval-small:      ## Gazebo, standard arena: make eval-small FIRST=1 N=3 WEIGHTS=...
	source $(ROS_SETUP) && SNR_FIRST=$(FIRST) python3 rl/eval_gazebo.py $(WEIGHTS) $$(( $(FIRST) + $(N) )) 0

eval-big:        ## Gazebo, big 8x6 m arena: make eval-big FIRST=51 N=3
	source $(ROS_SETUP) && SNR_ARENA=big SNR_FIRST=$(FIRST) python3 rl/eval_gazebo.py $(WEIGHTS) $$(( $(FIRST) + $(N) )) 0

eval-2d:         ## fast 2D evaluation on 100 layouts
	$(VENV_PY) rl/eval2d.py $(WEIGHTS) 100

train:           ## train a new policy: make train RUN=run11 STEPS=8e6 (about 2 h)
	$(VENV_PY) rl/train.py $(RUN) $(STEPS)
