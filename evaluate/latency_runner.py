import time
import logging
import statistics
from eksperimen.rag_agent_gemma import get_rag_agent
from evaluate.latency_logger import log_latency_per_query, save_latency_summary
from evaluate.benchmark_runner import TEST_CASES

logging.basicConfig(level=logging.INFO)


def run_latency_benchmark():
    agent = get_rag_agent()
    print(f"🚀 Memulai benchmark LATENSI End-to-End untuk model: {agent.model}")

    # Penampung data untuk dihitung rata-ratanya di akhir
    latencies = {
        "search": [],
        "rerank": [],
        "llm": [],
        "total": []
    }

    for i, case in enumerate(TEST_CASES, 1):
        q = case["q"]
        print(f"\n[{i}/{len(TEST_CASES)}] Mengukur Kueri: {q}")

        # 1. Mulai ukur total waktu
        start_time = time.perf_counter()

        # Eksekusi RAG
        resp = agent.answer(q)

        # Waktu selesai keseluruhan
        total_time = time.perf_counter() - start_time

        # 2. Ambil metrik internal dari response
        # (Wajib dipasang di dalam src/rag_agent.py, jika tidak ada, default ke 0.0)
        search_t = getattr(resp, 'search_time', 0.0)
        rerank_t = getattr(resp, 'rerank_time', 0.0)
        llm_t = getattr(resp, 'llm_time', 0.0)

        # 3. Catat per baris ke CSV
        log_latency_per_query(
            model_name=agent.model,
            question=q,
            search_time=search_t,
            rerank_time=rerank_t,
            llm_inference_time=llm_t,
            total_time=total_time
        )

        # Simpan ke memori untuk dihitung rata-rata akhir
        latencies["search"].append(search_t)
        latencies["rerank"].append(rerank_t)
        latencies["llm"].append(llm_t)
        latencies["total"].append(total_time)

        print(f"✅ Total: {total_time:.2f}s | Search: {search_t:.2f}s | Rerank: {rerank_t:.2f}s | LLM: {llm_t:.2f}s")

    # 4. Hitung Rata-rata dan Simpan ke JSON
    if len(latencies["total"]) > 0:
        summary = {
            "model": agent.model,
            "total_queries_tested": len(TEST_CASES),
            "average_latency_seconds": {
                "vector_search": round(statistics.mean(latencies["search"]), 3),
                "reranking": round(statistics.mean(latencies["rerank"]), 3),
                "llm_inference": round(statistics.mean(latencies["llm"]), 3),
                "total_end_to_end": round(statistics.mean(latencies["total"]), 3)
            }
        }
        save_latency_summary(agent.model, summary)

        print("\n📊 RATA-RATA LATENSI KESELURUHAN (Sesuai Tabel 5 Skripsi):")
        print(f"Enkoding Vektor & Search : {summary['average_latency_seconds']['vector_search']} detik")
        print(f"Reranking (Cross-Encoder): {summary['average_latency_seconds']['reranking']} detik")
        print(f"Inferensi LLM            : {summary['average_latency_seconds']['llm_inference']} detik")
        print(f"Total Latensi Rata-rata  : {summary['average_latency_seconds']['total_end_to_end']} detik")
        print("\nData riwayat tersimpan di: output/latency_results.csv")
        print("Data rata-rata tersimpan di: output/latency_summary_*.json")


if __name__ == "__main__":
    run_latency_benchmark()