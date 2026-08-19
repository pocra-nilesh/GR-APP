import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Page configuration
st.set_page_config(page_title="PoCRA GR AI App", layout="wide")
st.title("PoCRA's GR AI (TA DA Response only)")

# Configuration Paths - updated with your exact local embedding directory
MODEL_SNAPSHOT_PATH = "/home/pocraadmin/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"
EMBEDDING_MODEL_PATH = "/home/pocraadmin/GR/GR-APP/embedding_model"  # Update to your absolute path containing modules.json/model.safetensors
FAISS_INDEX_PATH = "GR_faiss_index"              # Update to your local FAISS folder path

@st.cache_resource
def load_rag_components():
    """Load local offline embedding model, FAISS index, and Gemma 4 31B snapshot model."""
    # 1. Load local embeddings offline & connect to FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"local_files_only": True}
    )

    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS vector store not found at '{FAISS_INDEX_PATH}'.")
        st.stop()

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Load Tokenizer & Model from exact snapshot path offline
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT_PATH,
        local_files_only=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SNAPSHOT_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )

    return retriever, tokenizer, model

with st.spinner("Loading offline vector store, local embeddings, and Gemma 4 31B weights..."):
    retriever, tokenizer, model = load_rag_components()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt handling
if query := st.chat_input("Ask something about your data..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching vector database & generating response..."):
            # Retrieve relevant context docs from FAISS
            source_docs = retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in source_docs])

            # Construct RAG augmented prompt template
            augmented_content = (
                f"Use the following pieces of context to answer the user's question.\n"
                f"If you don't know the answer, just say that you are yet to be trained to answer that question.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}"
            )

            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            chat_history.append({"role": "user", "content": augmented_content})

            formatted_prompt = tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )

            response_text = tokenizer.decode(
                outputs[0, inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )

            st.markdown(response_text)

            # Show retrieved sources expander
            with st.expander("Reference Sources from FAISS"):
                for i, doc in enumerate(source_docs):
                    st.write(f"**Source {i+1}:** {doc.page_content[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": response_text})
