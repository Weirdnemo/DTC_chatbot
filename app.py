import textwrap
import numpy as np
import pandas as pd
import google.generativeai as genai
from IPython.display import Markdown
import streamlit as st
import unicodedata
import pymupdf
from typing import List
from PyPDF2 import PdfReader

genai.configure(api_key="AIzaSyBrcc9Cir7L5uTC_99HxWoBYwsfHp8S2Qg")

def embed_fn(text, model='models/embedding-001', task_type="retrieval_document"):
    return genai.embed_content(model=model, content=text, task_type=task_type)["embedding"]


def upload_pdf_part(pdf_file):
    if pdf_file:
        pdf_reader = PdfReader(pdf_file)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    def is_pua(c):
        return unicodedata.category(c) == 'Co'

    final = "".join([char for char in text if not is_pua(char)])

    with open("content.txt", "w") as f:
        f.write(final)

    def word_splitter(source_text: str) -> List[str]:  
        import re  
        source_text = re.sub("\s+", " ", source_text)  # Replace multiple whitespaces  
        return re.split("\s", source_text)  # Split by single whitespace 

    def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float) -> List[str]:  
        text_words = word_splitter(text)  
        overlap_int = int(chunk_size * overlap_fraction)  
        chunks = []  
        for i in range(0, len(text_words), chunk_size):  
            chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]  
            chunk = " ".join(chunk_words)  
            chunks.append(chunk)  
        return chunks 

    chunks = get_chunks_fixed_size_with_overlap(final, 9999, overlap_fraction=0.2)

    Doc = []
    for i in chunks:
        doc_temp = {}
        doc_temp["Title"] = i.split()[0]
        doc_temp["content"] = i
        Doc.append(doc_temp)

    return(Doc)


# Streamlit app
st.title("Technovate Chatbot")
uploaded_file = st.file_uploader("Upload your PDF here:", type="pdf")

if uploaded_file is not None:
    list_doc = upload_pdf_part(uploaded_file)
    documents = list_doc
    df = pd.DataFrame(documents)
    df.columns = ['Title', 'Text']

    model = 'models/embedding-001'
    def embed_fn(title, text):
        return genai.embed_content(model=model,
                                    content=text,
                                    task_type="retrieval_document",
                                    title=title)["embedding"]

    df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)

    def find_best_passage(query, dataframe):
        query_embedding = genai.embed_content(model= 'models/embedding-001',
                                                content=query,
                                                task_type="retrieval_query")
        dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
        idx = np.argmax(dot_products)
        return dataframe.iloc[idx]['Text']

    def make_prompt(query, relevant_passage):
        escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = textwrap.dedent("""
            You are a chatbot for Delhi Technical Campus, answering questions for incoming students.
            Use the following passage to answer the question:

            QUESTION: '{query}'
            PASSAGE: '{escaped_passage}'

            ANSWER:
        """).format(query=query, escaped_passage=escaped_passage)
        return prompt

    model_name = 'models/gemini-1.5-flash'
    chat_model = genai.GenerativeModel(model_name)
    chat = chat_model.start_chat()

    # Initialize chat history (session state)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input and response generation
    if prompt := st.chat_input("Ask me about Delhi Technical Campus"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Find the relevant passage based on user query
        relevant_passage = find_best_passage(query=prompt, dataframe=df)

        # Create the prompt for the chat model
        chat_prompt = make_prompt(query=prompt, relevant_passage=relevant_passage)

        # Send the prompt to the chat model and get the response
        response = chat.send_message(chat_prompt).text

        # Display the assistant's response and the relevant passage
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant's response and relevant passage to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
