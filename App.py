import os
import dotenv
import gradio as gr
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load the document from the web
loader = WebBaseLoader("https://unctad.org/topic/competition-and-consumer-protection/un-guidelines-for-consumer-protection")
data = loader.load()

# Split the document text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Create the embedding model
embedding = OpenAIEmbeddings()

# Define the directory where embeddings will be stored
persist_dir = "Embeddings"

# Create the vector store and specify the persistence directory
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_dir)

# Load the vector store from disk
loaded_vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)

# Define the retriever
retriever = loaded_vectorstore.as_retriever(k=4)

# Create an instance of ChatMessageHistory to serve as memory
memory = ChatMessageHistory()

# Define the conversational retrieval chain with ChatGroq including memory
conversational_retrieval_chain = ChatGroq(
    temperature=0, 
    groq_api_key=GROQ_API_KEY, 
    model_name="llama3-8b-8192", 
    # memory=memory
)

# Function to handle the chatbot responses
def respond(message, chat_history):
    # Add user message to the chat history
    conversational_retrieval_chain.memory.add_user_message(message)
    
    # Get the response from the chain
    response = conversational_retrieval_chain.invoke({"messages": conversational_retrieval_chain.memory.messages})
    
    # Add the AI response to the chat history
    conversational_retrieval_chain.memory.add_ai_message(response['answer'])
    
    # Update chat history
    chat_history.append((message, response['answer']))
    return "", chat_history

# Create the Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    
    # Connect the submit action of the textbox to the respond function
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # Clear the chat history
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the interface
demo.launch(debug=True, share=True)
