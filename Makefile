PYTHON ?= $(CURDIR)/.venv/bin/python
OFFICIAL_REPO := $(CURDIR)/external/ml-cross-entropy
export PYTHONPATH := $(OFFICIAL_REPO):$(CURDIR)
export HF_HOME := $(CURDIR)/.cache/huggingface
export HF_TOKEN_PATH := $(HOME)/.cache/huggingface/token

.PHONY: environment correctness scaling filtering figure3 summaries plots validate analysis

environment:
	$(PYTHON) -m src.capture_environment

correctness:
	$(PYTHON) -m src.correctness

scaling:
	$(PYTHON) -m src.scaling

filtering:
	$(PYTHON) -m src.filtering

figure3:
	$(PYTHON) -m src.figure3_probabilities

summaries:
	$(PYTHON) -m src.summarize
	$(PYTHON) -m src.summarize_figure4

plots:
	$(PYTHON) -m src.plot_results

validate:
	$(PYTHON) -m src.validate_results

analysis: summaries plots validate
