import streamlit as st
import numpy as np

import os
import getpass

os.environ['OPENAI_API_KEY'] = "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"

from langchain_openai import ChatOpenAI

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQAWithSourcesChain

# loader = TextLoader("opt_str_new.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_store", OpenAIEmbeddings())

retriever = db.as_retriever()
model = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(),
                                                    chain_type="stuff",
                                                    retriever=retriever)

st.title("Optimal String")

# Initialize chat history
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display chat messages from history on app rerun
if "messages" in st.session_state:
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.write(message["content"])  # Change from st.markdown to st.write

# Display chat messages from history on app rerun

# React to user input
if prompt := st.chat_input("What is up?"):
  # Display user message in chat message container
  st.chat_message("user").markdown(prompt)
  # Add user message to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})

  question = prompt
  response = model({"question": question}, return_only_outputs=True)
  # print(response['answer'])
  # Display assistant response in chat message container
  with st.chat_message("assistant"):
    st.write(response['answer'])  # Change from st.markdown to st.write
  # Add assistant response to chat history
  st.session_state.messages.append({
      "role": "assistant",
      "content": response['answer']
  })

# while True:
#   prompt = st.text_input("Say something")
#   if prompt:
#     message = st.chat_message("user")
#     message.write(prompt)
#     question = prompt
#     response = model({"question": question}, return_only_outputs=True)
#     print(response['answer'])
#     message = st.chat_message("assistant")
#     message.write(response['answer'])
#     i = i + 1

#   else:
#     break
