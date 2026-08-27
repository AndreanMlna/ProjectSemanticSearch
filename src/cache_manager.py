"""
Cache Manager
Menyimpan hasil pencarian semantik untuk menghindari re-encoding query yang sama

Strategi:
- Cache key: hash dari (query_string + top_k) → unik per permintaan
- Cache value: hasil search lengkap dari ChromaDB
- TTL: configurable (default 60 menit)
- Eviction: LRU (Least Recently Used) saat cache penuh
"""

import hashlib
import time
import logging
from typing import Optional, Dict, Any
from collections import OrderedDict
from src.config import CACHE_MAX_SIZE, CACHE_TTL_MINUTES

logger = logging.getLogger("cache_manager")


# ═══════════════════════════════════════════════════════════════════
# CACHE ENTRY
# ═══════════════════════════════════════════════════════════════════

class CacheEntry:
    """Satu entry dalam cache beserta metadata waktu"""

    def __init__(self, data: Any, ttl_seconds: int):
        """
        Args:
            data: Data hasil search yang akan disimpan
            ttl_seconds: Berapa detik entry ini valid
        """
        self.data = data
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.hit_count = 0  # Berapa kali entry ini diakses

    def is_expired(self) -> bool:
        """Cek apakah entry sudah kedaluwarsa"""
        return time.time() > self.expires_at

    def access(self) -> Any:
        """Akses data dan catat hit"""
        self.hit_count += 1
        return self.data

    def time_remaining(self) -> float:
        """Sisa waktu valid dalam detik"""
        return max(0.0, self.expires_at - time.time())


# ═══════════════════════════════════════════════════════════════════
# SEARCH CACHE MANAGER
# ═══════════════════════════════════════════════════════════════════

class SearchCacheManager:
    """
    Cache untuk hasil pencarian semantik dokumen arsip.

    Menggunakan LRU (OrderedDict) untuk eviction saat cache penuh.
    Cache key dibuat dari hash query + top_k sehingga:
    - "surat keputusan" top_k=5 != "surat keputusan" top_k=10
    - Query yang sama persis akan mendapat hasil dari cache

    Contoh pemakaian:
        cache = SearchCacheManager(max_size=500, ttl_minutes=60)

        # Simpan hasil search
        cache.set(query="surat keputusan", top_k=5, results=data)

        # Ambil hasil dari cache (None jika tidak ada / expired)
        cached = cache.get(query="surat keputusan", top_k=5)
    """

    def __init__(self, max_size: int = 500, ttl_minutes: int = 60):
        """
        Args:
            max_size: Maksimal jumlah entry dalam cache (default 500)
            ttl_minutes: Lama cache valid dalam menit (default 60)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_minutes * 60
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Statistik
        self._total_hits = 0
        self._total_misses = 0
        self._total_evictions = 0

        logger.info(
            f"SearchCacheManager initialized: max_size={max_size}, ttl={ttl_minutes}min"
        )

    # ── Key Generation ───────────────────────────────────────────────

    @staticmethod
    def _make_key(query: str, top_k: int) -> str:
        """
        Buat cache key yang unik dari query dan top_k.

        Normalisasi query (lowercase + strip) sebelum hash
        supaya "Surat Keputusan" == "surat keputusan".

        Args:
            query: Query pencarian dari user
            top_k: Jumlah hasil yang diminta

        Returns:
            String hash MD5 sebagai cache key
        """
        normalized = f"{query.lower().strip()}|{top_k}"
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    # ── Core Operations ──────────────────────────────────────────────

    def get(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        """
        Ambil hasil search dari cache.

        Args:
            query: Query pencarian
            top_k: Jumlah hasil yang diminta

        Returns:
            Data hasil search jika ada dan belum expired, None jika tidak ada
        """
        key = self._make_key(query, top_k)

        if key not in self._cache:
            self._total_misses += 1
            logger.debug(f"Cache MISS: '{query}' (top_k={top_k})")
            return None

        entry = self._cache[key]

        # Hapus jika sudah expired
        if entry.is_expired():
            del self._cache[key]
            self._total_misses += 1
            logger.debug(f"Cache EXPIRED: '{query}' (top_k={top_k})")
            return None

        # LRU: pindahkan ke akhir (most recently used)
        self._cache.move_to_end(key)
        self._total_hits += 1

        logger.debug(
            f"Cache HIT: '{query}' (top_k={top_k}, "
            f"sisa {entry.time_remaining():.0f}s, hit_count={entry.hit_count + 1})"
        )

        return entry.access()

    def set(self, query: str, top_k: int, results: Dict[str, Any]) -> None:
        """
        Simpan hasil search ke cache.

        Jika cache penuh, hapus entry yang paling lama tidak diakses (LRU).

        Args:
            query: Query pencarian
            top_k: Jumlah hasil yang diminta
            results: Data hasil search dari ChromaDB yang akan dicache
        """
        key = self._make_key(query, top_k)

        # Update jika key sudah ada
        if key in self._cache:
            self._cache[key] = CacheEntry(results, self.ttl_seconds)
            self._cache.move_to_end(key)
            logger.debug(f"Cache UPDATED: '{query}' (top_k={top_k})")
            return

        # Evict LRU jika cache penuh
        if len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            self._total_evictions += 1
            logger.debug(f"Cache EVICTED LRU entry (key={evicted_key[:8]}...)")

        self._cache[key] = CacheEntry(results, self.ttl_seconds)
        logger.debug(f"Cache SET: '{query}' (top_k={top_k}, ttl={self.ttl_seconds}s)")

    def invalidate(self, query: str, top_k: int) -> bool:
        """
        Hapus satu entry dari cache secara manual.

        Berguna setelah upload/delete dokumen agar hasil cache
        tidak stale.

        Args:
            query: Query yang akan di-invalidate
            top_k: top_k yang sesuai

        Returns:
            True jika entry ditemukan dan dihapus, False jika tidak ada
        """
        key = self._make_key(query, top_k)
        if key in self._cache:
            del self._cache[key]
            logger.info(f"Cache INVALIDATED: '{query}' (top_k={top_k})")
            return True
        return False

    def clear(self) -> int:
        """
        Kosongkan seluruh cache.

        Dipanggil setelah bulk upload/delete agar cache tidak stale.

        Returns:
            Jumlah entry yang dihapus
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache CLEARED: {count} entries removed")
        return count

    def cleanup_expired(self) -> int:
        """
        Hapus semua entry yang sudah expired.

        Sebaiknya dipanggil secara periodik (misalnya setiap jam)
        untuk menjaga memori tetap efisien.

        Returns:
            Jumlah entry expired yang dihapus
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.info(f"Cache CLEANUP: {len(expired_keys)} expired entries removed")

        return len(expired_keys)

    # ── Statistics ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistik cache untuk endpoint /metrics dan /status.

        Returns:
            Dictionary berisi hit rate, jumlah entry, dll.
        """
        total_requests = self._total_hits + self._total_misses
        hit_rate = (
            round(self._total_hits / total_requests * 100, 2)
            if total_requests > 0
            else 0.0
        )

        # Hitung entry yang masih valid vs expired
        active_entries = sum(
            1 for e in self._cache.values() if not e.is_expired()
        )

        return {
            "enabled": True,
            "max_size": self.max_size,
            "current_size": len(self._cache),
            "active_entries": active_entries,
            "ttl_minutes": self.ttl_seconds // 60,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "total_evictions": self._total_evictions,
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SearchCacheManager("
            f"size={stats['current_size']}/{self.max_size}, "
            f"hit_rate={stats['hit_rate_percent']}%)"
        )


# ═══════════════════════════════════════════════════════════════════
# SINGLETON — dipakai oleh main_api.py dan routers
# ═══════════════════════════════════════════════════════════════════

_cache_instance: Optional[SearchCacheManager] = None


def get_cache_manager() -> SearchCacheManager:
    """
    Ambil instance global SearchCacheManager (singleton).
    Membaca konfigurasi dari src.config / environment variables.

    Returns:
        SearchCacheManager instance
    """
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = SearchCacheManager(
            max_size=CACHE_MAX_SIZE,
            ttl_minutes=CACHE_TTL_MINUTES
        )

    return _cache_instance


def reset_cache_manager() -> None:
    """Reset singleton (berguna untuk testing)"""
    global _cache_instance
    _cache_instance = None
