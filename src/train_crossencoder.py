import os
import json
import time
import random
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
OUTPUT_NAME = "crossencoder-arsip-finetuned"

# Konfigurasi Training
BATCH_SIZE = 16
EPOCHS = 10
SEEDS = [42, 123, 456, 789, 1024]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "indodoc")
BASE_OUTPUT_DIR = os.path.join(ROOT, "output")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CrossEncoderDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_cross_encoder_data(filename):
    file_path = os.path.join(DATA_DIR, filename)
    train_examples = []

    count_positive = 0
    count_negative = 0

    if not os.path.exists(file_path):
        print(f"[!] File {file_path} tidak ditemukan.")
        return []

    print(f"[*] Membaca data Cross-Encoder dari {filename}...")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                query = data.get("query", "").strip()
                document = data.get("document", "").strip()

                # DISESUAIKAN: Menggunakan float() agar sesuai dengan format 1.0 / 0.0 dari dataset baru
                label = float(data.get("label", 0.0))

                if not query or not document:
                    continue

                train_examples.append(InputExample(texts=[query, document], label=label))

                if label == 1.0:
                    count_positive += 1
                else:
                    count_negative += 1

            except json.JSONDecodeError:
                continue
            except ValueError:
                continue

    print(f"[*] Total data training: {len(train_examples)}")
    print(f"    - Pasangan Positif (Label 1.0)    : {count_positive}")
    print(f"    - Pasangan Negatif (Hard/Easy)    : {count_negative}")

    return train_examples


def train_reranker():
    train_examples_list = load_cross_encoder_data("train_cross_encoder.jsonl")

    if not train_examples_list:
        print("[!] Gagal: Data training kosong. Buat dataset hard-negatives terlebih dahulu.")
        return

    print(f"\n[INFO] Memulai Eksperimen Cross-Encoder dengan {len(SEEDS)} Seed Berbeda: {SEEDS}")

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"🚀 MULA TRAINING RERANKER UNTUK SEED: {seed}")
        print(f"{'=' * 60}")

        set_seed(seed)

        random.shuffle(train_examples_list)

        split_idx = int(len(train_examples_list) * 0.9)
        train_split = train_examples_list[:split_idx]
        val_split = train_examples_list[split_idx:]

        print(f"[*] Pembagian Data (Seed {seed}): {len(train_split)} Training | {len(val_split)} Validation")

        train_dataset = CrossEncoderDataset(train_split)

        # Ekstrak data untuk Evaluator
        val_pairs = [example.texts for example in val_split]
        val_labels = [example.label for example in val_split]

        evaluator = CrossEncoderClassificationEvaluator(
            sentence_pairs=val_pairs,
            labels=val_labels,
            name=f'indodoc-val-seed{seed}'
        )

        current_output_name = f"{OUTPUT_NAME}-seed-{seed}"
        current_output_dir = os.path.join(BASE_OUTPUT_DIR, current_output_name)
        os.makedirs(current_output_dir, exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=BATCH_SIZE
        )

        print(f"[2] Initializing Cross-Encoder Model: {MODEL_NAME} (Seed: {seed})")

        # PERBAIKAN: Menangkap exception secara spesifik (koneksi/offline/OS error)
        try:
            model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=384)
        except (OSError, RuntimeError, ConnectionError) as e:
            print(f"[!] Gagal terhubung ke Hugging Face ({str(e)}). Mencoba memuat dari cache lokal...")
            model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=384, local_files_only=True)

        warmup_steps = int(len(train_loader) * EPOCHS * 0.1)

        print(f"[3] Mulai Training Cross-Encoder selama {EPOCHS} epochs...")
        start_time = time.time()

        model.fit(
            train_dataloader=train_loader,
            evaluator=evaluator,
            epochs=EPOCHS,
            warmup_steps=warmup_steps,
            output_path=current_output_dir,
            save_best_model=True,
            show_progress_bar=True,
            use_amp=True
        )

        end_time = time.time()
        duration = end_time - start_time

        print("[*] Menyimpan model secara eksplisit (Forced Save)...")
        model.save(current_output_dir)

        print(f"\n[DONE] Training Seed {seed} Selesai dalam {duration:.2f} detik")

        stats_path = os.path.join(current_output_dir, "training_stats.json")
        stats_data = {
            "model_name": current_output_name,
            "seed": seed,
            "training_duration_seconds": duration,
            "duration_minutes": duration / 60,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "total_samples": len(train_examples_list),
            "train_samples": len(train_split),
            "val_samples": len(val_split)
        }

        with open(stats_path, "w") as f:
            json.dump(stats_data, f, indent=4)

        print(f"[*] Model optimal dan stats tersimpan di: {current_output_dir}\n")

    print(f"\n{'=' * 60}")
    print(f"🎉 SEMUA EKSPERIMEN FINE-TUNING CROSS-ENCODER SELESAI!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_reranker()