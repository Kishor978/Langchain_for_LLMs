from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS, Annoy
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import  create_retriever_tool
from langchain_community.llms import Ollama
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
import streamlit as st

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wikipedia=WikipediaQueryRun(api_wrapper=api_wrapper)
wikipedia.name


arxiv_wrapper=ArxivAPIWrapper(top_K_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
arxiv.name
tools=[wikipedia,arxiv]
llm=Ollama(model="llama2")

# prompt=hub.pull("hwchase17/openai-functions-agent")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries."),
        ("user","Question:{question}"),
        ("system","Answer:{answer}"),
        
    ]
)

agent=create_openai_tools_agent(llm,tools,prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

st.title("Chatbot using LLama-2")
input_text=st.text_input("Search for a question")
if input_text:
    agent_executor.invoke({"question":input_text})
