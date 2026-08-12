from huggingface_hub import snapshot_download

# Fetches the entire 31B instruction-tuned model repository 
# This model file size is ~58GB
model_path = snapshot_download(
    repo_id="google/gemma-4-31B-it",
    ignore_patterns=["*.msgpack", "*.h5"] # Optional: skips non-PyTorch weight formats to save space
)

print(f"Download complete! Model files are saved at: {model_path}")


#import torch
#from transformers import AutoProcessor, AutoModelForMultimodalLM

# Define the model ID exactly as you did during download
#model_id = "google/gemma-4-31B-it"

#print("Loading processor from local cache...")
#processor = AutoProcessor.from_pretrained(
#    model_id, 
#    local_files_only=True
#)

#print("Loading model weights from local cache into memory...")
#model = AutoModelForMultimodalLM.from_pretrained(
#    model_id,
#    torch_dtype=torch.bfloat16,
#    device_map="auto",
#    local_files_only=True  # Bypasses internet check completely
#)

#print("Gemma 4-31B loaded successfully from local files!")
