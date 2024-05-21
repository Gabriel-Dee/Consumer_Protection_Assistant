# Import necessary libraries
import os 
import dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# Load environment variables
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Setup the chat model
Model = 'llama3-8b-8192'
chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=Model)

# Define the prompt templates
system_prompt = "You are a helpful assistant."
human_prompt = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])

# Build the conversational chain
chain = prompt | chat

# Create the embedding model
embedding = OpenAIEmbeddings()

# Define the persistence directory
persist_dir = 'Embeddings'

# Load the vector store
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)

# Define the retriever
retriever = vectorstore.as_retriever(k=4)

# Define the query transforming retriever chain
query_transform_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.")
])

query_transforming_retriever_chain = RunnableBranch(
    (lambda x: len(x.get("messages", [])) == 1,
        (lambda x: x["messages"][-1].content) | retriever,
    ),
    query_transform_prompt | chat | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")

# Define the question answering prompt
question_answering_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="messages")
])

# Create the document chain
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

# Combine the chains into the conversational retrieval chain
conversational_retrieval_chain = RunnablePassthrough.assign(
    context=query_transforming_retriever_chain,
).assign(
    answer=document_chain,
)

# Initialize Streamlit app
st.title("Conversational Chatbot")
st.write("Ask me anything about the Guidelines for Consumer Protection")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Chat input and message display
if prompt := st.chat_input("You: "):
    # Add user message to chat history
    st.session_state.chat_history.add_user_message(prompt)
    
    # Invoke the conversational retrieval chain
    response = conversational_retrieval_chain.invoke({"messages": st.session_state.chat_history.messages})
    
    # Add AI message to chat history
    st.session_state.chat_history.add_ai_message(response["answer"])

# Display chat messages
for msg in st.session_state.chat_history.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)