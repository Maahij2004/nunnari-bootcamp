import sys
import os

# Add the current directory to Python's search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from core.graph import graph # Ensure this matches your folder case!


st.set_page_config(page_title="NotebookLM Replica", layout="wide")

# --- UI Layout ---
st.title("📓 Local NotebookLM Replica")

col1, col2 = st.columns([0.7, 0.3])

with st.sidebar:
    st.header("Files & Settings")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")
    
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with col1:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about your documents or the web..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the Agentic Graph
        with st.spinner("Thinking..."):
            inputs = {"query": prompt}
            result = graph.invoke(inputs)
            answer = result["response"]

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with col2:
    st.header("📌 Saved Notes")
    # Display saved markdown notes from storage/notes
    import os
    notes = os.listdir("storage/notes") if os.path.exists("storage/notes") else []
    for note in notes:
        st.info(f"📄 {note}")