import logging
from typing import Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from src.config import CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME
from src.logging_utils import setup_logging

logger = setup_logging("chroma_client")

_chroma_client: Optional[ClientAPI] = None
_collection: Optional[Collection] = None


def get_chroma_client() -> ClientAPI:
    """Mengembalikan instance HttpClient ChromaDB singleton."""
    global _chroma_client
    if _chroma_client is None:
        init_chroma_client()
    return _chroma_client


def init_chroma_client() -> ClientAPI:
    """Menginisialisasi koneksi ke server ChromaDB."""
    global _chroma_client, _collection
    logger.info(f"Connecting to ChromaDB Server at http://{CHROMA_HOST}:{CHROMA_PORT} ...")
    try:
        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✅ ChromaDB client initialized. Collection '{COLLECTION_NAME}' ready.")
        return _chroma_client
    except Exception as e:
        logger.error(f"❌ Failed to connect to ChromaDB at http://{CHROMA_HOST}:{CHROMA_PORT}: {e}")
        _chroma_client = None
        _collection = None
        raise e


def get_collection() -> Optional[Collection]:
    """Mengembalikan koleksi aktif ChromaDB."""
    global _collection, _chroma_client
    if _collection is None:
        try:
            if _chroma_client is None:
                init_chroma_client()
            else:
                _collection = _chroma_client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            logger.warning(f"Could not retrieve ChromaDB collection '{COLLECTION_NAME}': {e}")
            return None
    return _collection


def reset_collection() -> Optional[Collection]:
    """Menghapus dan membuat ulang koleksi ChromaDB."""
    global _collection, _chroma_client
    client = get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' deleted.")
    except Exception:
        pass
    _collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Collection '{COLLECTION_NAME}' recreated.")
    return _collection
