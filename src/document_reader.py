
import os
import logging
import re
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("document_reader")

# Batas karakter teks yang dikirim ke LLM
MAX_CHARS_PER_DOC = 3000
# Threshold panjang dokumen — di atas batas ini dianggap "panjang"
LONG_DOC_THRESHOLD = 3000


def extract_text_from_pdf(file_path: str) -> str:
    """Mengekstrak teks dari file PDF menggunakan pypdf."""
    try:
        import pypdf
        text_parts = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
        result = "\n".join(text_parts)
        logger.debug(f"PDF extracted: {len(result)} chars from {os.path.basename(file_path)}")
        return result
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {e}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """Mengekstrak teks dari file Word (.docx) menggunakan python-docx."""
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        result = "\n".join(paragraphs)
        logger.debug(f"Docx extracted: {len(result)} chars from {os.path.basename(file_path)}")
        return result
    except Exception as e:
        logger.error(f"Docx extraction failed for {file_path}: {e}")
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """Membaca teks dari file .txt biasa."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            result = f.read()
        logger.debug(f"Txt read: {len(result)} chars from {os.path.basename(file_path)}")
        return result
    except Exception as e:
        logger.error(f"Txt read failed for {file_path}: {e}")
        return ""


def extract_full_text(file_path: str) -> str:
    """Router utama — mengekstrak teks berdasarkan ekstensi file."""
    if not os.path.exists(file_path):
        # Biarkan return kosong agar sistem memicu fallback ke Vector DB
        return ""

    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        logger.warning(f"Format file tidak didukung: {ext}")
        return ""



def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Memotong teks panjang menjadi chunk-chunk yang memiliki overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            boundary = text.rfind(".", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    logger.debug(f"Text chunked: {len(text)} chars -> {len(chunks)} chunks")
    return chunks


def find_most_relevant_chunk(query: str, chunks: List[str], top_n: int = 2) -> str:

    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    query_words = set(re.findall(r'\w+', query.lower()))
    if not query_words:
        return "\n\n".join(chunks[:top_n])

    scored = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        matches = sum(1 for w in query_words if w in chunk_words)
        score = matches / len(query_words)
        scored.append((score, i, chunk))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    top_scored = scored[:top_n]
    top_scored.sort(key=lambda x: x[1])

    top_chunks = [chunk for _, _, chunk in top_scored]
    return "\n\n".join(top_chunks)


# FUNGSI FILTER UTAMA TEKS FISIK

def get_document_context(file_path: str, query: str, max_chars: int = MAX_CHARS_PER_DOC) -> str:

    full_text = extract_full_text(file_path)

    if not full_text:
        return ""

    full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()

    if len(full_text) <= LONG_DOC_THRESHOLD:
        context = full_text[:max_chars]
        logger.debug(f"Short physical doc: {len(context)} chars used")
        return context

    chunks = chunk_text(full_text, chunk_size=800, overlap=150)
    relevant = find_most_relevant_chunk(query=query, chunks=chunks, top_n=3)
    context = relevant[:max_chars]
    logger.debug(f"Long physical doc: {len(chunks)} chunks, {len(context)} chars used")
    return context


# INTERFAS UTAMA (DIPANGGIL OLEH rag_agent.py)

def get_context_for_results(
    search_results: List[Any],
    query: str,
    upload_dir: str,
    max_chars_per_doc: int = MAX_CHARS_PER_DOC
) -> List[Dict[str, Any]]:

    enriched = []

    for result in search_results:
        is_dict = isinstance(result, dict)
        file_name = result.get("file_name") if is_dict else getattr(result, "file_name", None)
        title = result.get("title") if is_dict else getattr(result, "title", "Untitled")
        snippet = result.get("snippet") if is_dict else getattr(result, "snippet", "")
        score = result.get("score") if is_dict else getattr(result, "score", 0.0)
        download_url = result.get("download_url") if is_dict else getattr(result, "download_url", "")

        if not file_name:
            logger.warning("Menemukan hasil search tanpa file_name, dilewati.")
            continue

        file_path = os.path.join(upload_dir, file_name)

        full_context = get_document_context(
            file_path=file_path,
            query=query,
            max_chars=max_chars_per_doc
        )

        if not full_context:

            db_content = result.get("document_asli") if is_dict else getattr(result, "document_asli", None)

            if not db_content:
                db_content = result.get("content_only") if is_dict else getattr(result, "content_only", None)

            if db_content:
                full_context = db_content[:max_chars_per_doc]
                logger.info(f"Sync Sukses: Menggunakan teks utuh dari Vector DB untuk {file_name}")
            else:
                full_context = snippet
                logger.warning(f"Emergency Fallback: Menggunakan 'snippet' pendek untuk {file_name}")

        enriched.append({
            "title": title,
            "file_name": file_name,
            "score": score,
            "download_url": download_url,
            "snippet": snippet,
            "full_context": full_context,
            "context_length": len(full_context)
        })

        logger.info(
            f"Context enriched: '{title}' "
            f"-> {len(full_context)} chars "
            f"({'File Fisik' if os.path.exists(file_path) else 'Vector Store Data_Only'})"
        )

    return enriched