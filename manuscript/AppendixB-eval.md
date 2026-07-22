# Appendix B. Evaluation without LangSmith

LangSmith is the LangChain company's paid platform for LLM observability, evaluation, and prompt versioning. It is genuinely good software, and if you are already paying for it and it fits your workflow, none of this appendix will change your mind. What this appendix is about is: how to evaluate an LLM pipeline when you have chosen *not* to pay for LangSmith and you want to stay entirely in open source.

The good news is that evaluation without LangSmith is not a step backward. Every technique that matters can be built with a few dozen lines of Python plus, optionally, one of several open source evaluation tools. Nothing in the LangSmith interface is magic.

## The two things evaluation actually is

Most "evaluation" writing conflates two very different activities:

- **Reference-based evaluation**: you have a set of `(input, expected_output)` pairs and you want to know how often your pipeline produces the expected output. This is exactly what unit tests are; the twist is that "matches" is fuzzy because LLM output varies.
- **Reference-free evaluation**: you have inputs but no expected outputs, and you want to know whether the pipeline's answer is good. This is closer to code review than unit testing. You either have a human read the output or you have a stronger LLM read it.

The techniques below cover both.

## The LLM-as-judge pattern

By far the most useful reference-free evaluation technique. You ask a strong model to score a weaker model's output. The prompt looks like:

```text
You are evaluating an answer to a question.

Question: {question}
Answer: {answer}

Rate the answer on a scale of 1 to 5, where:
1 = incorrect or nonsensical
5 = fully correct, well-explained, appropriately detailed

Reply with only the number.
```

Ten lines of Python around a `.structured_predict(Score)` call (with Score being a Pydantic model with a `int` field constrained to 1-5) gives you a numerical score. Run it over a batch of test inputs; look at the distribution. Anything meaningfully below the mean is worth investigating by hand.

Two rules for LLM-as-judge to work:

- **Use a stronger model for the judge than the one being evaluated.** A model can't reliably grade something at its own capability level.
- **Include the criteria in the prompt.** "Rate on quality" is too vague. "Rate on factual accuracy, given that the following sources are the ground truth: ..." is specific enough to be useful.

## Reference-based evaluation in three lines

For reference-based evaluation on structured outputs, plain `assert` works:

```python
result = agent.invoke({"messages": [HumanMessage(content=case["input"])]})
answer = result["messages"][-1].content
assert case["expected_keyword"].lower() in answer.lower(), f"Missed on {case['id']}"
```

That is a real test. Put it in `pytest` and run it in CI. For structured outputs it is even easier: Pydantic validation handles the schema check, and you assert on specific fields:

```python
person = llm.structured_predict(Person, prompt, input_text=case["text"])
assert person.name == case["expected_name"]
assert person.email == case["expected_email"]
```

I have shipped whole evaluation suites that were nothing but a few dozen of these.

## When you want a real evaluation UI

If you want traces, per-run drill-downs, dashboards, and the whole "observability platform" experience, three open source projects handle this well:

- **[Langfuse](https://langfuse.com/)**: closest in feel to LangSmith. Self-hosted (Docker Compose or Kubernetes), integrates with LangChain, LlamaIndex, and the OpenAI SDK. Free for local use; paid tiers for hosted. Their comparison page against LangSmith is fair and worth reading.
- **[MLflow](https://mlflow.org/)**: the classic ML experiment tracker. Has picked up LLM-specific features in the last two years. Best if you already have MLflow infrastructure from traditional ML work.
- **[Phoenix / Arize OpenInference](https://github.com/Arize-ai/phoenix)**: trace collector built on the OpenInference OpenTelemetry standard. Both LangChain and LlamaIndex can emit OpenInference traces, and Phoenix visualizes them locally. This is my go-to for "give me visibility into what an agent is doing" when I don't want a full observability platform.

For a solo developer, Phoenix + a folder of Python evaluation scripts covers 90% of what a full LangSmith subscription would give you, at a cost of some initial setup and zero recurring bill.

## Practical minimum

If none of the above appeals, here is the smallest viable evaluation setup for a serious project:

1. A `evals/` directory in your repo.
2. A JSON or YAML file of test cases: `[{"id": "...", "input": "...", "expected": "..."}, ...]`.
3. A single Python script that iterates the cases, runs each one through your pipeline, computes a match (exact, substring, or LLM-judge, whichever makes sense), and prints a table.
4. A `make eval` or `uv run eval.py` command in CI or a pre-release checklist.

That is enough to catch regressions. Every larger evaluation setup grows out of this seed.
