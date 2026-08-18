import os
import json
import time
import random
import re
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.modules import Transformer, Pooling

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_NAME = "minilm-dokumen-arsip-boosted-seed-42"

BATCH_SIZE = 16
EPOCHS = 10
MAX_SEQ_LENGTH = 384
SEED = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "indodoc")
BASE_OUTPUT_DIR = os.path.join(ROOT, "output")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, OUTPUT_NAME)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrackedMNRLoss(MultipleNegativesRankingLoss):
    def __init__(self, model):
        super().__init__(model)
        self.loss_history = []

    def forward(self, sentence_features, labels):

        loss_val = super().forward(sentence_features, labels)


        self.loss_history.append(loss_val.item())

        return loss_val


class ExamplesDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_augmented_data(filename):
    file_path = os.path.join(DATA_DIR, filename)
    examples = []

    if not os.path.exists(file_path):
        print(f"[!] File {file_path} tidak ditemukan.")
        return []

    print(f"[*] Membaca & Meng-augmentasi data dari {filename}...")


    count_title_content = 0
    count_snippet_title = 0
    count_keyword_title = 0

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
                    count_title_content += 1

                    content_snippet = content[:200]
                    if len(content_snippet) > 50:
                        examples.append(InputExample(texts=[content_snippet, title]))
                        count_snippet_title += 1

                    if "kata kunci" in content.lower():

                        parts = re.split(r'(?i)kata kunci:?', content)
                        if len(parts) > 1:

                            keywords = parts[-1].strip().rstrip('.')
                            if keywords:
                                examples.append(InputExample(texts=[keywords, title]))
                                count_keyword_title += 1

            except json.JSONDecodeError:
                continue

    print(f"[*] Total data setelah augmentasi: {len(examples)} (Naik berkali lipat!)")
    print(f"    - Title dengan Content              : {count_title_content} pasang")
    print(f"    - Title dengan Content Snippet      : {count_snippet_title} pasang")
    print(f"    - Content (Kata Kunci) dengan Title : {count_keyword_title} pasang")

    return examples


def train_boosted():
    # 1. Load Data Augmented
    train_examples_list = load_augmented_data("train.jsonl")

    if not train_examples_list:
        print("[!] Gagal: Data training kosong.")
        return

    train_dataset = ExamplesDataset(train_examples_list)

    print(f"\n{'=' * 60}")
    print(f"🚀 MULAI TRAINING FINAL (BEST MODEL) - SEED: {SEED}")
    print(f"{'=' * 60}")

    # Set seed agar shuffle dan bobot konsisten
    set_seed(SEED)

    # Buat output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    # 2. Inisialisasi Model
    print(f"[2] Initializing Model: {MODEL_NAME} (Seed: {SEED})")
    word_embedding_model = Transformer(MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH)

    pooling_model = Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 3. Loss Function
    train_loss = TrackedMNRLoss(model)

    print(f"[*] Model berjalan di device: {model.device}")

    # 4. Training Process
    warmup_steps = int(len(train_loader) * EPOCHS * 0.1)

    print(f"[3] Mulai Training BOOSTED (Seed {SEED}) selama {EPOCHS} epochs...")
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

    print(f"\n[DONE] Training Seed {SEED} Selesai dalam {duration:.2f} detik")

    # Simpan Statistik Waktu
    stats_path = os.path.join(OUTPUT_DIR, "training_stats.json")
    stats_data = {
        "model_name": OUTPUT_NAME,
        "seed": SEED,
        "training_duration_seconds": duration,
        "duration_minutes": duration / 60,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "samples_count": len(train_examples_list)
    }

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=4)

    # Simpan Riwayat Loss
    loss_path = os.path.join(OUTPUT_DIR, "loss_history.json")
    with open(loss_path, "w") as f:
        json.dump({
            "keterangan": f"Nilai loss per batch selama proses training untuk seed {SEED}",
            "loss_per_batch": train_loss.loss_history
        }, f, indent=4)

    print(f"[*] Model final otomatis telah tersimpan oleh fit() di {OUTPUT_DIR}...")
    print(f"[*] Data riwayat Loss dan stats tersimpan di: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    train_boosted()