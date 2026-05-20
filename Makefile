.PHONY: check-env test

PYTHON ?= python

check-env:
	$(PYTHON) scripts/00_check_env.py

test:
	$(PYTHON) -m pytest -q
