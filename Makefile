PYTHON ?= python

.PHONY: install prepare train compare ablation test lint serve docker-build

install:
	$(PYTHON) -m pip install -e ".[dev]"

prepare:
	$(PYTHON) -m recsys.cli prepare

train:
	$(PYTHON) -m recsys.cli train --model deepfm

compare:
	$(PYTHON) -m recsys.cli compare

ablation:
	$(PYTHON) -m recsys.cli ablation

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

serve:
	$(PYTHON) -m recsys.cli serve --port 8000

docker-build:
	docker build -t ctr-deepfm:latest .
