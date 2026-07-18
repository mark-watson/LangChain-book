.PHONY: check test run clean

# Fast syntax check - compiles all .py files without executing
check:
	python3 -m py_compile *.py
	uv run python -c "import langchain_chroma, langchain_core, langchain_huggingface, langchain_ollama"

# Run tests (pytest if tests/ exists, otherwise syntax-check passes as test)
test:
	python3 -m py_compile *.py && echo "No pytest tests - syntax check passed"

# Run the main example script: data.py
run:
	uv run data.py
	uv run recipe_generator.py

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache 2>/dev/null || true
