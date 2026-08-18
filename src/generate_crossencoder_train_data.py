import os
import json
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(ROOT, "data", "indodoc", "train.jsonl")
OUTPUT_FILE = os.path.join(ROOT, "data", "indodoc", "train_cross_encoder.jsonl")


NATURAL_TEMPLATES = [
    "Tolong carikan informasi tentang {}",
    "Bagaimana prosedur atau SOP mengenai {}",
    "Apa yang dimaksud dengan {}",
    "Berikan saya dokumen mengenai {}",
    "Saya butuh dokumen terkait {}",
    "Tolong carikan penetapan atau surat keputusan terkait {}",
    "Ada arsip untuk {}?",
    "Saya ingin melihat arsip tentang {}",
    "Di mana saya bisa membaca {}",
    "Tolong tampilkan dokumentasi saat {}"
]


def calculate_word_overlap(text1, text2):
    """Menghitung jumlah kata yang sama untuk mencari Hard Negatives"""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    return len(set1.intersection(set2))


def generate_crossencoder_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"[!] File tidak ditemukan: {INPUT_FILE}")
        return

    documents_data = []

    print(f"[*] Membaca data dan mengekstrak variasi kueri natural dari {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                title = data.get("title", "").strip()
                content = data.get("content", "").strip()

                if title and content:
                    full_document = f"Judul: {title}\nIsi: {content}"

                    # 1. Gunakan Judul yang DIBUNGKUS Bahasa Natural (Perbaikan list literal di sini)
                    queries = [random.choice(NATURAL_TEMPLATES).format(title.lower())]

                    # Sesekali masukkan judul mentah (20% kemungkinan) agar model tetap pintar keyword
                    if random.random() > 0.8:
                        queries.append(title)

                    # 2. Menggunakan potongan isi (snippet) sebagai kueri (diubah sedikit agar seperti pertanyaan)
                    snippet = content[:100]
                    if len(snippet) > 50:
                        queries.append(f"Apakah ada dokumen yang isinya membahas: {snippet}...")

                    # 3. Menggunakan kata kunci yang DIBUNGKUS Bahasa Natural
                    if "kata kunci" in content.lower():
                        parts = re.split(r'(?i)kata kunci:?', content)
                        if len(parts) > 1:
                            keywords = parts[-1].strip().rstrip('.')
                            if keywords:
                                # Bersihkan koma agar lebih luwes saat dibaca model
                                clean_kws = keywords.replace(",", " ")
                                queries.append(random.choice(NATURAL_TEMPLATES).format(clean_kws))
                                if random.random() > 0.8:
                                    queries.append(keywords)

                    documents_data.append({
                        "queries": queries,
                        "full_document": full_document
                    })
            except json.JSONDecodeError:
                continue

    cross_encoder_data = []
    total_docs = len(documents_data)

    print(f"[*] Total dokumen asli: {total_docs}")
    print("[*] Merakit pasangan data Positif (1.0) dan Negatif (0.0)...")

    for i in range(total_docs):
        current_data = documents_data[i]
        correct_document = current_data["full_document"]
        other_docs = [doc["full_document"] for j, doc in enumerate(documents_data) if j != i]

        for query in current_data["queries"]:
            # [1] Pasangan Positif -> Kueri + Dokumen Benar
            cross_encoder_data.append({
                "query": query,
                "document": correct_document,
                "label": 1.0
            })

            # [2] Pasangan Hard Negative -> Cari dokumen lain yang kata-katanya paling mirip
            other_docs.sort(key=lambda x: calculate_word_overlap(query, x), reverse=True)
            hard_negative = other_docs[0]
            cross_encoder_data.append({
                "query": query,
                "document": hard_negative,
                "label": 0.0
            })

            # [3] Pasangan Easy Negative -> Ambil acak dari sisa dokumen
            easy_negative = random.choice(other_docs[5:])
            cross_encoder_data.append({
                "query": query,
                "document": easy_negative,
                "label": 0.0
            })

    random.shuffle(cross_encoder_data)

    print(f"[*] Menyimpan {len(cross_encoder_data)} pasangan data ke {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for item in cross_encoder_data:
            out_f.write(json.dumps(item) + "\n")

    print("[+] Dataset Cross-Encoder BERHASIL dibuat!")
    pos_count = sum(1 for x in cross_encoder_data if x["label"] == 1.0)
    neg_count = sum(1 for x in cross_encoder_data if x["label"] == 0.0)
    print(f"    - Label 1.0 (Dokumen Tepat)  : {pos_count} baris")
    print(f"    - Label 0.0 (Dokumen Salah)  : {neg_count} baris")
    print(f"    - Total Dataset Pelatihan    : {len(cross_encoder_data)} baris")


if __name__ == "__main__":
    generate_crossencoder_dataset()