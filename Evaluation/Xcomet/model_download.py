from comet import download_model, load_from_checkpoint
from huggingface_hub import login
login()

model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)

model.save_pretrained("Xcomet-XL-model")