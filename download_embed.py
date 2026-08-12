from sentence_transformers import SentenceTransformer

# Define your desired model (e.g., all-MiniLM-L6-v2)
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# This line downloads and loads the model
model = SentenceTransformer(model_name)

# (Optional) Save it permanently to a specific local folder
model.save("./embedding_model")