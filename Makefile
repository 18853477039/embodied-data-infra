# Developer entry points. Run `make help` for the target list.
PY ?= .venv/bin/python
UV ?= uv

COLLECT_CONFIG ?= configs/collect/gym_hil_pick_cube.json
CLEAN_CONFIG ?= configs/clean/pick_cube.yaml
TRAIN_CONFIG ?= configs/train/sac_gym_hil_pick_cube.json
CURATED_ROOT ?= data/curated
DATASET ?=

.PHONY: help venv install test lint format check collect clean convert validate train pipeline

help:
	@printf "Targets:\n"
	@printf "  venv      Create the local virtual environment\n"
	@printf "  install   Install the package with dev extras into .venv\n"
	@printf "  test      Run the test suite\n"
	@printf "  lint      Check formatting and lint rules\n"
	@printf "  format    Apply ruff fixes and formatting\n"
	@printf "  check     Run lint + tests (CI equivalent)\n"
	@printf "  collect   Record episodes (CONFIG=$(COLLECT_CONFIG))\n"
	@printf "  clean     Raw -> cleaned layer (CONFIG=$(CLEAN_CONFIG))\n"
	@printf "  convert   Cleaned layer -> canonical LeRobot dataset\n"
	@printf "  validate  Re-validate a curated dataset (DATASET=...)\n"
	@printf "  train     Train on the newest curated dataset\n"
	@printf "  pipeline  clean + convert\n"

venv:
	$(UV) venv --python 3.12 .venv

install: venv
	$(UV) pip install -e ".[dev]" --python $(PY)

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

check: lint test

collect:
	$(PY) -m embodied_data_infra.cli collect --config $(COLLECT_CONFIG)

clean:
	$(PY) -m embodied_data_infra.cli clean --config $(CLEAN_CONFIG)

convert:
	$(PY) -m embodied_data_infra.cli convert --config $(CLEAN_CONFIG)

validate:
	@test -n "$(DATASET)" || (printf "DATASET=/path/to/dataset is required\n"; exit 2)
	$(PY) -m embodied_data_infra.cli validate --config $(CLEAN_CONFIG) --dataset $(DATASET)

train:
	$(PY) -m embodied_data_infra.cli train --config $(TRAIN_CONFIG) \
		--latest-curated --curated-root $(CURATED_ROOT)

pipeline: clean convert
