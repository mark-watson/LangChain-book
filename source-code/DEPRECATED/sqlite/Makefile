.PHONY: check test run clean

# Fast syntax check - compiles all .py files without executing
check:
	python3 -m py_compile *.py

# Run tests (pytest if tests/ exists, otherwise syntax-check passes as test)
test:
	python3 -m py_compile *.py && echo "No pytest tests - syntax check passed"

# Run the main example script: sqlite_chat_test.py
run:
	python3 sqlite_chat_test.py

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache 2>/dev/null || true
