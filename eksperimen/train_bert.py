# src/train_bert.py
import os
import json
import time
import torch  # Ditambahkan untuk mengecek CUDA
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample

# IMPORT YANG BENAR (Sesuai dengan sentence-transformers v5.4+)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.modules import Transformer, Pooling

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OUTPUT_NAME = "sbert-mpnet-dokumen-arsip"

BATCH_SIZE = 8
EPOCHS = 4
MAX_SEQ_LENGTH = 256

# --- PATHS (SAMA) ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "indodoc")
OUTPUT_DIR = os.path.join(ROOT, "output", OUTPUT_NAME)


class ExamplesDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_archive_data(filename):
    file_path = os.path.join(DATA_DIR, filename)
    examples = []
    if not os.path.exists(file_path):
        print(f"[!] File {file_path} tidak ditemukan.")
        return []

    print(f"[*] Membaca data training dari {filename}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "title" in data and "content" in data:
                    title = data["title"].strip()
                    content = data["content"].strip()
                    if not title or not content:
                        continue
                    examples.append(InputExample(texts=[title, content]))
            except json.JSONDecodeError:
                continue
    print(f"[*] Total pasangan data latih: {len(examples)}")
    return examples


def train_and_measure():
    # 1. Load Data
    train_examples_list = load_archive_data("train.jsonl")
    if not train_examples_list: return

    train_dataset = ExamplesDataset(train_examples_list)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    # 2. Inisialisasi Model (Sama, tapi memuat MPNet)
    print(f"[2] Initializing Model: {MODEL_NAME}")

    # PERBAIKAN: Menggunakan kelas Transformer secara langsung
    word_embedding_model = Transformer(MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH)

    # PERBAIKAN: Menggunakan kelas Pooling secara langsung
    pooling_model = Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )

    # PENGECEKAN HARDWARE EKSPLISIT UNTUK CUDA (GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Pengecekan Hardware: PyTorch akan menggunakan -> {device.upper()}")

    # Memasukkan device ke dalam inisialisasi model
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    # PERBAIKAN: Menggunakan MultipleNegativesRankingLoss secara langsung
    train_loss = MultipleNegativesRankingLoss(model)

    print(f"[*] Model berjalan di device: {model.device}")

    # 3. Training
    warmup_steps = int(len(train_loader) * EPOCHS * 0.1)

    print(f"[3] Mulai Training {OUTPUT_NAME}...")
    start_time = time.time()

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        show_progress_bar=True,
        use_amp=True
    )

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n[DONE] Training Selesai dalam {duration:.2f} detik")

    # --- TAMBAHAN 1: SIMPAN MODEL EKSPLISIT ---
    print(f"[*] Menyimpan model final ke {OUTPUT_DIR}...")
    model.save(OUTPUT_DIR)

    # 4. Benchmark Latency (Pasti lebih lambat dari MiniLM)
    print("\n[4] Benchmark Kecepatan (Inference Speed)...")
    dummy_text = "Surat Keputusan Rektor tentang Penetapan Beasiswa Mahasiswa Berprestasi"
    model.encode(dummy_text)  # Pemanasan

    t_start = time.time()
    for _ in range(100):
        model.encode(dummy_text)
    t_end = time.time()

    avg_latency = (t_end - t_start) / 100
    print(f"Model: {MODEL_NAME}")
    print(f"Latency: {avg_latency * 1000:.2f} ms/kalimat")

    # --- TAMBAHAN 2: SIMPAN STATISTIK WAKTU KE JSON ---
    stats_path = os.path.join(OUTPUT_DIR, "training_stats.json")
    stats_data = {
        "model_name": OUTPUT_NAME,
        "base_model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "samples": len(train_examples_list),
        "duration_seconds": duration,
        "duration_minutes": duration / 60,
        "avg_latency_ms": avg_latency * 1000
    }

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=4)

    print(f"[*] Statistik waktu & teknis tersimpan di: {stats_path}")


if __name__ == "__main__":
    train_and_measure()