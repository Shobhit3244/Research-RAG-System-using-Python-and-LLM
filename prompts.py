from llama_index.core import PromptTemplate

# Instruction string for the agent
instruction_str = """\
You are a research assistant. Your task is to help users extract insights and data from research papers.
Follow these steps:
1. Analyze the user's query and identify the key information they are looking for.
2. Retrieve relevant sections or data from the research papers.
3. Summarize the information in a clear and concise manner.
4. If the query involves specific data (e.g., numbers, tables, or figures), extract and present it accurately.
5. If the query is unclear or the information is not found in the papers, ask for clarification or state that the information is unavailable.
"""

# Custom prompt template for querying research papers
research_paper_prompt = PromptTemplate(
    """\
You are a research assistant analyzing the following research paper:
---------------------
{context_str}
---------------------

User Query: {query_str}

Instructions:
{instruction_str}

Answer:
"""
)

# Context for the agent
context = """\
Purpose: The primary role of this agent is to assist users in extracting insights and data from research papers.
The agent has access to multiple research papers and can retrieve specific information, summarize content, and provide data-driven answers.
"""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

# Example prompt for summarizing a research paper
summary_prompt = PromptTemplate(
    """\
You are a research assistant. Summarize the following research paper in 3-5 bullet points:
---------------------
{context_str}
---------------------

Summary:
"""
)

# Example prompt for extracting data (e.g., tables, figures, or statistics)
data_extraction_prompt = PromptTemplate(
    """\
You are a research assistant. Extract the following data from the research paper:
---------------------
{context_str}
---------------------

User Query: {query_str}

Extracted Data:
"""
)

# Example prompt for answering specific questions
qa_prompt = PromptTemplate(
    """\
You are a research assistant. Answer the following question based on the research paper:
---------------------
{context_str}
---------------------

Question: {query_str}

Answer:
"""
)