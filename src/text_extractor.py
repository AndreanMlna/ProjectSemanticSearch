import os
from pypdf import PdfReader
import docx


def extract_text_from_file(file_path):
    """
    Fungsi pintar untuk mendeteksi jenis file
    dan mengambil teks di dalamnya.
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            return _read_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return _read_docx(file_path)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return ""  # File tidak dikenali, kembalikan kosong
    except Exception as e:
        print(f"[!] Gagal baca file {file_path}: {e}")
        return ""


def _read_pdf(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def _read_docx(path):
    doc = docx.Document(path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)