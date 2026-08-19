import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from difflib import SequenceMatcher

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RerankerModule")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = os.getenv("CE_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
MIN_SCORE_THRESHOLD = 0.20


def sigmoid(x: Any) -> Any:
    return 1 / (1 + np.exp(-x))


def fuzzy_keyword_match(keyword: str, text_to_check: str, threshold: float = 0.75) -> bool:
    if keyword in text_to_check:
        return True

    words_in_text = text_to_check.split()
    for word in words_in_text:
        clean_word = word.strip("?,.!\"()'")
        if len(clean_word) > 3:
            if SequenceMatcher(None, keyword, clean_word).ratio() >= threshold:
                return True
    return False


class ArsirReranker:
    _instance: Optional['ArsirReranker'] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ArsirReranker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name_or_path: Optional[str] = None):
        if self._initialized:
            return

        model_path = model_name_or_path or DEFAULT_MODEL
        logger.info(f"[*] Menginisialisasi Cross-Encoder Reranker (HuggingFace): {model_path}")
        try:
            self.model = CrossEncoder(model_path)
            factory = StopWordRemoverFactory()
            self.stopword_remover = factory.create_stop_word_remover()
            self._initialized = True
            logger.info("[+] Cross-Encoder Reranker dan Sastrawi berhasil dimuat.")
        except Exception as e:
            logger.error(f"[!] Gagal memuat model Reranker: {str(e)}")
            raise e

    def rerank(self, query: str, chroma_results: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if not chroma_results or not chroma_results.get('documents') or not chroma_results['documents'][0]:
            logger.warning("[-] Tidak ada kandidat dokumen dari ChromaDB untuk di-rerank.")
            return []

        documents: List[str] = chroma_results['documents'][0]
        metadatas: List[Dict[str, Any]] = chroma_results['metadatas'][0]
        distances: List[float] = chroma_results.get('distances', [[0.0] * len(documents)])[0]
        ids: List[str] = chroma_results['ids'][0] if 'ids' in chroma_results else [f"doc_{i}" for i in
                                                                                   range(len(documents))]

        logger.info(f"[*] Memulai proses hybrid reranking untuk {len(documents)} kandidat dokumen...")

        pair_inputs = []
        for i, doc_text in enumerate(documents):
            title = metadatas[i].get('title', '')

            full_context_to_score = f"Judul: {title}\nIsi: {doc_text}"
            pair_inputs.append([query, full_context_to_score])

        try:
            raw_scores = self.model.predict(pair_inputs)
            rerank_scores = sigmoid(raw_scores)
        except Exception as e:
            logger.error(f"[!] Terjadi kesalahan saat komputasi reranking: {str(e)}")
            return [{"id": ids[i], "metadata": metadatas[i], "score": 0.0} for i in range(min(top_k, len(documents)))]

        cleaned_query = self.stopword_remover.remove(query.lower())
        query_keywords = list(set([
            w.strip("?,.!\"()'") for w in cleaned_query.split()
            if len(w.strip("?,.!\"()'")) > 3
        ]))

        logger.info(f"[*] Kata kunci hasil filter Sastrawi: {query_keywords}")

        reranked_list = []
        alpha = 0.6
        beta = 0.4

        for i in range(len(documents)):

            chroma_similarity = 1 / (1 + float(distances[i]))
            reranker_score = float(rerank_scores[i])
            final_hybrid_score = (alpha * chroma_similarity) + (beta * reranker_score)

            if query_keywords:
                title = metadatas[i].get('title', '')

                full_text = documents[i]
                text_to_check = f"{title} {full_text}".lower()

                matched_words = sum(
                    1 for kw in query_keywords if fuzzy_keyword_match(kw, text_to_check, threshold=0.75))
                match_ratio = matched_words / len(query_keywords)

                if match_ratio < 0.25:
                    penalty = 0.10 * (1.0 - match_ratio)
                    final_hybrid_score -= penalty
                    final_hybrid_score = max(0.0, final_hybrid_score)

            reranked_list.append({
                "id": ids[i],
                "score": final_hybrid_score,
                "title": metadatas[i].get('title', 'Tanpa Judul'),

                "snippet": metadatas[i].get('snippet', ''),
                "file_name": metadatas[i].get('file_name', '-'),
                "download_url": metadatas[i].get('download_url', '')
            })

        reranked_list.sort(key=lambda x: x["score"], reverse=True)

        # Filter dokumen yang skornya di bawah threshold
        filtered_results = [doc for doc in reranked_list if doc["score"] >= MIN_SCORE_THRESHOLD]
        final_results = filtered_results[:top_k]

        if final_results:
            logger.info(
                f"[+] Rerank selesai. Top doc: '{final_results[0]['title']}' (Skor: {final_results[0]['score']:.4f})")
        else:
            logger.warning("[-] Tidak ada dokumen yang lolos batas minimal threshold.")

        return final_results


_reranker_instance = None


def get_reranker(model_name_or_path: Optional[str] = None) -> ArsirReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = ArsirReranker(model_name_or_path=model_name_or_path)
    return _reranker_instance