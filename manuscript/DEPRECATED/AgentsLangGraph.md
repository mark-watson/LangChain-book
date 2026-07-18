# Experimenting with Agents Using LangGraph and Pluggable Search Tools

Dear reader, you have probably spent much of your working life engaging in agent-based work flows with you and your co-workers acting as cooperating agents to set goals for projects, talk to potential users of systems you will build or users who will read documentation you produce as a team.

Here we keep a human in the loop (you!) and augment your capabilities with agents. This work flow is often referred to as “agentic workflows” but here we use the term “agent.” Many of us now have at least two years experience experience using LLMs with simple workflows:

- Ask an LLM general questions about the world (subject to hallucinations).
- Describe a new project and ask for a plan based on innate knowledge encoded in an LLM.
- Describe a new project and ask for a plan based both on web search results and on innate knowledge encoded in an LLM.
- Describe a problem and ask an LLM to generate code in a specific programming language.
- Ask an LLM to summarize lengthy articles, research papers, or books.

We want to expand the utility of LLMs in our work and private lives by building on these simple workflows, adding agents who can research, verify results, and work more as partners with us.

## Tool Use Helps Reduce Hallucinations and Facilitates More Robust Software Systems Using LLMs

For our purposes here, we will explore three frameworks for using web search:

- Brave Search APIs
- Tavaly Search APIs
- Duckduckgo Search APIs

I also use Azure and Google search APIs in my work, but will not cover those here.

In practice, it might be best to choose a single search API framework but here I want to give you options.

We will create an abstraction over search APIs and services, implementing in a few lines of code a query function of the form:

```python
  def query(prompt):
    ...
    return [(uri1, title1, description1), ...]
```
