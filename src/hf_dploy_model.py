import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

load_dotenv()

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError("HF_TOKEN tidak ditemukan. Pastikan file .env ada dan token sudah terisi.")

print("[*] Melakukan login ke Hugging Face...")
login(token=TOKEN)

# Path Absolut folder model lokal
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_model_lokal = os.path.join(ROOT, "output", "minilm-dokumen-arsip-boosted-new-seed-42")

print(f"[*] Memeriksa folder model di: {folder_model_lokal}")
if not os.path.exists(folder_model_lokal):
    raise FileNotFoundError(f"Folder model tidak ditemukan di {folder_model_lokal}")

print("[*] Memuat model ke memori...")
model = SentenceTransformer(folder_model_lokal)

nama_repo_tujuan = "andrerean/minilm-arsip-kampus-v1"
print(f"[*] Mengunggah SELURUH file model ke {nama_repo_tujuan}...")
print("[*] Mohon tunggu, proses upload file model (safetensors) memakan waktu beberapa saat...")

# Melakukan push dengan menyertakan commit_message yang jelas
model.push_to_hub(
    nama_repo_tujuan,
    private=False,
    commit_message="Upload full fine-tuned model weights and tokenizers"
)

print("[V] Upload Selesai! Model Anda sekarang sudah benar-benar mengudara.")