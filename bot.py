pip install streamlit transformers
import streamlit as st
from transformers import pipeline
import base64

# Load Q&A model pipeline
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

# Function to set the background image (world map)
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/BlankMap-World6.svg/2000px-BlankMap-World6.svg.png");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Set background
add_bg_from_url()

# Title of the app
st.title("Ask Anything About Geography")

# Prompt for user input
question = st.text_input("Enter your geography-related question:")

# Process the question using the Q&A model
if st.button("Ask"):
    if question:
        context = """
        Geography is the study of places and the relationships between people and their environments. 
        Geographers explore both the physical properties of Earth's surface and the human societies spread across it. 
        They also examine how human culture interacts with the natural environment and the way that locations and places can impact people.
        The field of geography is broad, ranging from the study of physical landscapes, climates, and ecosystems to the analysis of political and economic geographies.
        """
        # Get the answer
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']

        # Show the answer
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please enter a question.")


