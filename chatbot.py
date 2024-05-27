import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

dotenv.load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

Model = 'llama3-8b-8192'
chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=Model)

system_prompt = "You are a helpful assistant."
human_prompt = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])

chain = prompt | chat

embedding = OpenAIEmbeddings()

persist_dir = 'Embeddings'

vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)

retriever = vectorstore.as_retriever(k=4)

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

question_answering_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context. If the context is not relevant to the user's question, respond with 'I'm sorry, I can only answer questions related to the provided context.'\n\n{context}"),
    MessagesPlaceholder(variable_name="messages")
])

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

conversational_retrieval_chain = RunnablePassthrough.assign(
    context=query_transforming_retriever_chain,
).assign(
    answer=document_chain,
)

def get_chatbot_response(user_message, chat_history):
    chat_history.add_user_message(user_message)
    response = conversational_retrieval_chain.invoke({"messages": chat_history.messages})
    chat_history.add_ai_message(response["answer"])
    return response["answer"], chat_history
