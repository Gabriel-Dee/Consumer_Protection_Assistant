import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Define Streamlit sidebar for API key input
with st.sidebar:
    GROQ_API_KEY = st.text_input("LangChain API Key", type="password")
    "[Get a LangChain API key](https://langchain.io)"

# Initialize LangChain ChatGroq model
chat = None
if GROQ_API_KEY:
    Model = 'llama3-8b-8192'
    chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=Model)

# Define Streamlit app layout
st.title("LangChain Chatbot")
st.caption("💬 A Streamlit chatbot powered by LangChain")

# Define function to handle user input and generate chatbot response
def generate_response(input_text):
    if chat:
        response = chat.invoke([HumanMessage(content=input_text)])
        return response[0].content
    else:
        return "Please enter your LangChain API key to enable the chatbot."

# Get user input
user_input = st.text_input("You:", "")

# Check if user has entered input
if user_input:
    # Generate response
    bot_response = generate_response(user_input)
    # Display bot response
    st.text_area("Bot:", value=bot_response, height=100)
