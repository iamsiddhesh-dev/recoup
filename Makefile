# make is a thin wrapper over `python -m recoup`, which is the canonical entry
# point. make is not installed by default on Windows, where this is developed;
# the wrapper exists so the documented `make demo` works for anyone cloning on
# Linux or macOS.

PY ?= python

.PHONY: setup demo eval reproduce test lint clean

setup:  ## install the package and dev dependencies
	$(PY) -m pip install -e ".[dev]"

demo:  ## run a batch end to end and serve the control room
	$(PY) -m recoup demo

eval:  ## print the arms table: gross, incremental, cost, net, refusals
	$(PY) -m recoup eval

reproduce:  ## regenerate every committed figure from fixed seeds
	$(PY) -m recoup reproduce

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

clean:
	$(PY) -m recoup clean
