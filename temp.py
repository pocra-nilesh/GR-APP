import fitz  # PyMuPDF
import json
import os
from vector_store import create_vector_db, load_db
import tempfile


def extract_marathi_pdf(pdf_path, output_txt_path):
    # Open the digital PDF file
    doc = fitz.open(pdf_path)
    
    # Open output file with UTF-8 encoding to support Marathi script
    with open(output_txt_path, "w", encoding="utf-8") as text_file:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract plain text
            text = page.get_text()
            
            # Write page header and extracted content
            text_file.write(f"--- Page {page_num + 1} ---\n")
            text_file.write(text + "\n\n")
            
    print(f"Extraction successful! Saved to: {output_txt_path}")


def get_pdf_metadata(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Retrieve the metadata dictionary
    metadata = doc.metadata
    
    # Print the metadata cleanly using JSON formatting
    print("--- PDF Metadata ---")
    print(json.dumps(metadata, indent=4, ensure_ascii=False))
    
    return metadata


# Run the function
#pdf_data = get_pdf_metadata("your_marathi_file.pdf")      
#extract_marathi_pdf("your_marathi_file.pdf", "extracted_marathi_text.txt")


BASE = r"C:\Work\GR AI\GR's"
#print("Folders in the GR's Folder",os.listdir(BASE))


TA_PATH = BASE + r"\TA DA"

FILES = []
for i, doc in enumerate(os.listdir(TA_PATH)):
    print("Document:", doc)

    if i == 3:
        break
    else:
        FILES.append(TA_PATH + f'\{doc}')  
        create_vector_db(FILES)

                            