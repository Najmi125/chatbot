import streamlit as st
from transformers import pipeline

# Load the model for question-answering
@st.cache_resource
def load_qa_model():
    return pipeline("text-generation", model="gpt2")

# Load the model once when the app starts
qa_model = load_qa_model()

# Custom CSS for a colorful background and fun style
st.markdown(
    """
    <style>
    body {
        background-color: #e6f7ff;
    }
    .main {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput > div > input {
        background-color: #ffe6f0;
        color: #333;
        border-radius: 5px;
    }
    .stButton > button {
        background-color: #66ccff;
        color: white;
        border-radius: 5px;
    }
    h1 {
        color: #ff6666;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    p {
        color: #333;
        font-size: 18px;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Ask Azb any general knowledge question!")
st.write("Hello kids! Type your question below, and I'll do my best to answer with fun and facts!")

# Get user input
user_question = st.text_input("What's your question? 🤔")

if user_question:
    with st.spinner('Thinking... 💭'):
        response = qa_model(user_question, max_length=100, num_return_sequences=1)
        st.write("Azb's Answer:")
        st.write(response[0]['generated_text'])
