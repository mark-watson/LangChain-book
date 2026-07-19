.PHONY: check test run clean subdirs

# Directories with Python code (excludes data/, data_small/, prompt_examples/)
SUBDIRS = \
	cooking_recipes \
	embedchain_test \
	extraction \
	from_langchain_docs \
	google_drive_llm \
	hugging_face \
	kg_agent \
	kor \
	langchain_getting_started \
	langgraph_durable \
	langgraph_fundamentals \
	langgraph_hitl \
	langgraph_react_agent \
	langgraph_supervisor \
	llama_index_agent \
	llama_index_deploy \
	llama_index_extract \
	llama_index_indices \
	llama_index_ingest \
	llama_index_intro \
	llama_index_rerank \
	llama_index_router \
	llama_index_workflows \
	local_search \
	ollama_langchain \
	rag \
	rag_langchain \
	sql_agent \
	summarization \
	tavily_search \
	text_db_llm \
	tool_search_math_example \
	tools_langchain

# Run 'check' (fast syntax check) in every subdirectory
check:
	@for dir in $(SUBDIRS); do \
		echo "=== $$dir ===" ; \
		$(MAKE) -C $$dir check || exit 1 ; \
	done

# Run 'test' in every subdirectory
test:
	@for dir in $(SUBDIRS); do \
		echo "=== $$dir ===" ; \
		$(MAKE) -C $$dir test || exit 1 ; \
	done

# Run 'run' (main example) in every subdirectory
run:
	@for dir in $(SUBDIRS); do \
		echo "=== $$dir ===" ; \
		$(MAKE) -C $$dir run || exit 1 ; \
	done

# Clean build artifacts in every subdirectory and top level
clean:
	rm -rf */__pycache__
	rm -rf */.venv
	@for dir in $(SUBDIRS); do \
		echo "=== $$dir ===" ; \
		$(MAKE) -C $$dir clean ; \
	done
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache 2>/dev/null || true
