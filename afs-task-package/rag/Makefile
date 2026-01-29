# Makefile for RAG System
# Provides convenient commands for development and deployment

.PHONY: help install clean build query health metrics example test lint format

help:
	@echo "RAG System - Available Commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Clean generated files"
	@echo "  make build      - Build the RAG pipeline"
	@echo "  make rebuild    - Force rebuild the pipeline"
	@echo "  make query      - Run example query"
	@echo "  make health     - Run health check"
	@echo "  make metrics    - Show system metrics"
	@echo "  make example    - Run example script"
	@echo "  make lint       - Run code quality checks"
	@echo "  make format     - Format code with black"
	@echo "  make test       - Run tests"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ src/__pycache__ src/*/__pycache__
	rm -rf .pytest_cache .mypy_cache
	rm -rf logs/*.log
	rm -rf faiss_index
	@echo "✓ Cleaned"

build:
	@echo "Building RAG pipeline..."
	python main.py build

rebuild:
	@echo "Force rebuilding RAG pipeline..."
	python main.py build --force-rebuild

query:
	@echo "Running example query..."
	python main.py query "What is the exam policy?" --summary "Student asking about rules"

health:
	@echo "Running health check..."
	python main.py health

metrics:
	@echo "Getting system metrics..."
	python main.py metrics

example:
	@echo "Running example script..."
	python example.py

lint:
	@echo "Running linters..."
	@echo "Note: Install flake8 and mypy first (pip install flake8 mypy)"
	-flake8 src/ --max-line-length=100
	-mypy src/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	@echo "Note: Install black first (pip install black)"
	-black src/ main.py example.py

test:
	@echo "Running tests..."
	@echo "Note: Install pytest first (pip install pytest)"
	-pytest tests/ -v
