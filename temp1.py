import os
from vector_store import create_vector_db
from pathlib import Path

FOLDER = r"C:\Work\GR AI\Docs"

for doc in os.listdir(FOLDER):
    print("Creating Knowledge Base for:", Path(FOLDER) / doc)
    create_vector_db([Path(FOLDER) / doc])

print("Knowledge base created successfully!")

