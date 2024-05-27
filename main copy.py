import os
import dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
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
    ("system", "Answer the user's questions based on the below context. If the context is not relevant to the user's question, respond with 'I'm sorry, I can only answer questions related to the provided context.'\n\n{context}"),
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

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    history: list

@app.post("/chat/")
async def chat(request: ChatRequest):
    # Create chat history from request
    chat_history = ChatMessageHistory(messages=[HumanMessage(content=msg) if idx % 2 == 0 else HumanMessage(content=msg, is_user=False) for idx, msg in enumerate(request.history)])

    # Add user message to chat history
    chat_history.add_user_message(request.message)
    
    # Invoke the conversational retrieval chain
    response = conversational_retrieval_chain.invoke({"messages": chat_history.messages})
    
    # Add AI message to chat history
    chat_history.add_ai_message(response["answer"])

    # Prepare response
    response_history = [msg.content for msg in chat_history.messages]

    return {"history": response_history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
