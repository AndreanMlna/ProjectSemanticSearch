import os
from sentence_transformers.cross_encoder import CrossEncoder

# Setup path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
OUTPUT_DIR = os.path.join(ROOT, "output", "crossencoder-base-model")


def download_and_save_model():
    print(f"[*] Bersiap mengunduh model: {MODEL_NAME} dari HuggingFace...")
    print(f"[*] Mohon tunggu, proses ini membutuhkan koneksi internet yang stabil.\n")

    try:
        # Mengunduh model dari Hugging Face
        model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=384)
        print("\n[+] Model berhasil diunduh ke RAM!")

        # Menyimpan model secara lokal
        print(f"[*] Menyimpan model ke folder lokal: {OUTPUT_DIR} ...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save(OUTPUT_DIR)

        print("\n[✅] SUKSES! Model base berhasil disimpan.")
        print(f"Path lokal Anda: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n[❌] Terjadi kesalahan saat mengunduh/menyimpan: {str(e)}")


if __name__ == "__main__":
    download_and_save_model()