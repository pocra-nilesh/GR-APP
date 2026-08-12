import os
import tempfile

import streamlit as st
from openai import OpenAI

from vector_store import create_vector_db, load_db


# ----------------------------------------
# Streamlit Config
# ----------------------------------------

st.set_page_config(
    page_title="LM Studio RAG",
    layout="wide"
)

st.title("Document RAG")

# ----------------------------------------
# Sidebar
# ----------------------------------------

st.sidebar.header("Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT / MD",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

if st.sidebar.button("Build Knowledge Base"):

    if not uploaded_files:
        st.sidebar.warning("Upload at least one document.")
        st.stop()

    paths = []

    for uploaded in uploaded_files:

        suffix = os.path.splitext(uploaded.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(uploaded.read())
            paths.append(tmp.name)

    with st.spinner("Creating FAISS Index..."):

        create_vector_db(paths)

    st.sidebar.success("Knowledge Base Created")

# ----------------------------------------
# Load Vector Store
# ----------------------------------------

db = None
retriever = None

if os.path.exists("faiss_index"):

    db = load_db()

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

# ----------------------------------------
# LM Studio Client
# ----------------------------------------

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# Automatically detect the loaded model

try:
    models = client.models.list()
    MODEL_NAME = models.data[0].id
except Exception:
    MODEL_NAME = "local-model"

# ----------------------------------------
# Chat
# ----------------------------------------

st.header("Ask Questions")

query = st.text_input(
    "Question",
    placeholder="Ask something about the uploaded documents..."
)

if st.button("Ask"):

    if db is None:

        st.error("Please build the knowledge base first.")

    elif not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents..."):

            docs = retriever.invoke(query)
            
            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            prompt = f"""
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
reply exactly:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{query}
"""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

        st.subheader("Answer")

        st.write(answer)

        st.divider()

        st.subheader("Retrieved Context")

        for i, doc in enumerate(docs, start=1):

            with st.expander(f"Chunk {i}"):

                st.write(doc.page_content)

                st.caption(doc.metadata)