from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import QEngines
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine

load_dotenv()

tools = [
    note_engine,
].extend(QEngines)

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while(prompt := input("Enter a prompt (or q to Quit): ")).lower() != "q":
    result = agent.query(prompt)
    print(result)
