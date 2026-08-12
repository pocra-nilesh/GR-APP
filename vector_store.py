from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from markitdown import MarkItDown



#EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "GR_faiss_index"

# Path to the folder where you saved the model files
local_model_path = "./embedding_model"

# Load the model directly from your hard drive
#embeddings = SentenceTransformer(local_model_path)
embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path
)

# Single converter for PDFs, DOCX, TXT, and MD files
md_converter = MarkItDown()


def load_document(path):
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    # Supported extensions for MarkItDown
    supported_extensions = [".pdf", ".docx", ".txt", ".md"]

    if suffix in supported_extensions:
        try:
            # MarkItDown extracts layout, tables, and text as clean Markdown
            result = md_converter.convert(str(path_obj))
            
            return [
                Document(
                    page_content=result.text_content, 
                    metadata={"source": str(path_obj), "file_type": suffix}
                )
            ]
        except Exception as e:
            raise Exception(f"Failed to convert {path_obj.name}: {str(e)}")
    else:
        raise Exception(f"Unsupported file type: {suffix}")


def create_vector_db(files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=400
    )

    documents = []

    for file in files:
        docs = load_document(file)
        chunks = splitter.split_documents(docs)
        documents.extend(chunks)

    db = FAISS.from_documents(
        documents = documents,
        embedding = embeddings
    )

    db.save_local(DB_DIR)

    return db


def load_db():
    return FAISS.load_local(
        DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
