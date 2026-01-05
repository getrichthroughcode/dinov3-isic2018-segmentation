.PHONY: help install clean train evaluate visualize test lint format

# Variables
PYTHON := python
PIP := pip
DATA_DIR := data
RUNS_DIR := runs
RESULTS_DIR := results

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make prepare-data     - Prepare ISIC2018 dataset"
	@echo "  make train-all        - Train all models (all data regimes)"
	@echo "  make evaluate-all     - Evaluate all trained models"
	@echo "  make visualize-all    - Generate all visualizations"
	@echo "  make clean            - Remove generated files"
	@echo "  make test             - Run unit tests"
	@echo "  make lint             - Run code linters"
	@echo "  make format           - Format code with black"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

prepare-data:
	$(PYTHON) scripts/prepare_data.py --data-dir $(DATA_DIR) --output-dir $(DATA_DIR)/processed

# Training shortcuts
train-baseline-25:
	$(PYTHON) scripts/train.py --model baseline --data-fraction 0.25 --output-dir $(RUNS_DIR)/baseline_25_percent

train-baseline-50:
	$(PYTHON) scripts/train.py --model baseline --data-fraction 0.5 --output-dir $(RUNS_DIR)/baseline_50_percent

train-baseline-100:
	$(PYTHON) scripts/train.py --model baseline --data-fraction 1.0 --output-dir $(RUNS_DIR)/baseline_100_percent

train-dinov3s-25:
	$(PYTHON) scripts/train.py --model dinov3_small --data-fraction 0.25 --output-dir $(RUNS_DIR)/dinov3s_unet_25_percent

train-dinov3s-50:
	$(PYTHON) scripts/train.py --model dinov3_small --data-fraction 0.5 --output-dir $(RUNS_DIR)/dinov3s_unet_50_percent

train-dinov3s-100:
	$(PYTHON) scripts/train.py --model dinov3_small --data-fraction 1.0 --output-dir $(RUNS_DIR)/dinov3s_unet_100_percent

train-dinov3b-25:
	$(PYTHON) scripts/train.py --model dinov3_base --data-fraction 0.25 --output-dir $(RUNS_DIR)/dinov3b_unet_25_percent

train-dinov3b-50:
	$(PYTHON) scripts/train.py --model dinov3_base --data-fraction 0.5 --output-dir $(RUNS_DIR)/dinov3b_unet_50_percent

train-dinov3b-100:
	$(PYTHON) scripts/train.py --model dinov3_base --data-fraction 1.0 --output-dir $(RUNS_DIR)/dinov3b_unet_100_percent

train-dinov3l-25:
	$(PYTHON) scripts/train.py --model dinov3_large --data-fraction 0.25 --output-dir $(RUNS_DIR)/dinov3l_unet_25_percent

train-dinov3l-50:
	$(PYTHON) scripts/train.py --model dinov3_large --data-fraction 0.5 --output-dir $(RUNS_DIR)/dinov3l_unet_50_percent

train-dinov3l-100:
	$(PYTHON) scripts/train.py --model dinov3_large --data-fraction 1.0 --output-dir $(RUNS_DIR)/dinov3l_unet_100_percent

train-all: train-baseline-25 train-baseline-50 train-baseline-100 \
           train-dinov3s-25 train-dinov3s-50 train-dinov3s-100 \
           train-dinov3b-25 train-dinov3b-50 train-dinov3b-100 \
           train-dinov3l-25 train-dinov3l-50 train-dinov3l-100

# Evaluation
evaluate-all:
	@for model_dir in $(RUNS_DIR)/*/; do \
		if [ -f "$$model_dir/best.pt" ]; then \
			echo "Evaluating $$model_dir"; \
			$(PYTHON) scripts/evaluate.py --model-path $$model_dir/best.pt --output-dir $(RESULTS_DIR)/$$(basename $$model_dir); \
		fi \
	done

# Visualization
visualize-all:
	$(PYTHON) scripts/visualize_samples.py --models-dir $(RUNS_DIR) --output-dir assets

# Testing and code quality
test:
	pytest tests/ -v

lint:
	flake8 src/ scripts/ tests/
	mypy src/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/

clean-runs:
	rm -rf $(RUNS_DIR)/*

clean-results:
	rm -rf $(RESULTS_DIR)/*

clean-all: clean clean-runs clean-results
	@echo "All generated files cleaned"
