.PHONY: setup data check sdd-check baseline train evaluate infer test lint clean

PYTHON ?= python3.11
MODEL ?= Qwen/Qwen3-0.6B-Base
ADAPTER ?= artifacts/qwen3-0.6b-ticket-lora

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"

data:
	PYTHONPATH=src $(PYTHON) -m sft_sop.build_data

check:
	PYTHONPATH=src $(PYTHON) -m sft_sop.check_data

sdd-check:
	PYTHONPATH=src $(PYTHON) -m sft_sop.check_sdd

baseline:
	PYTHONPATH=src $(PYTHON) -m sft_sop.evaluate \
		--model $(MODEL) \
		--split test \
		--report reports/baseline.json

train:
	PYTHONPATH=src $(PYTHON) -m sft_sop.train --config configs/sft_lora.yaml

evaluate:
	PYTHONPATH=src $(PYTHON) -m sft_sop.evaluate \
		--model $(MODEL) \
		--adapter $(ADAPTER) \
		--split test \
		--report reports/finetuned.json

infer:
	PYTHONPATH=src $(PYTHON) -m sft_sop.infer \
		--model $(MODEL) \
		--adapter $(ADAPTER) \
		--text "退款已经等了七天，今天下班前必须到账，否则我要投诉"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m ruff check src tests

clean:
	@echo "为避免误删 checkpoint，请手动确认后删除 artifacts/ 和 reports/ 下的结果。"
