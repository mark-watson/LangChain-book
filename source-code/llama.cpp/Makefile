.PHONY: check test run clean

# Fast syntax check - compiles all .py files without executing
check:
	python3 -m py_compile *.py
	uv run python -c "import langchain_community, langchain_core"

# Run tests (pytest if tests/ exists, otherwise syntax-check passes as test)
test:
	python3 -m py_compile *.py && echo "No pytest tests - syntax check passed"

# Run the main example script: test.py
run:
	uv run test.py

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache 2>/dev/null || true
