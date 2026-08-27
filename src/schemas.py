from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Kueri pencarian teks", min_length=1)
    top_k: int = Field(
        10, description="Jumlah dokumen teratas yang ingin dikembalikan", ge=1, le=50
    )


class UpdateDocumentRequest(BaseModel):
    title: str | None = Field(None, description="Judul baru dokumen")
    content: str | None = Field(None, description="Isi/deskripsi baru dokumen")
    keywords: str | None = Field(None, description="Kata kunci baru dokumen")
