import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"


print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


print("\nLoading BGE-M3...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)

print("Model loaded!")

texts = [
    "Hiiii.. am not an imposter tho 🫣",
    "You are an imposter fs ;)",
    "bro are you free tonight?",
    "kal kya kar raha hai?"
]


print("\nCreating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=4,
    show_progress_bar=True,
    convert_to_tensor=True
)


print("\n===== RESULT =====")

print("Shape:", embeddings.shape)

print("Device:", embeddings.device)

print("First vector:")
print(embeddings[0][:10])