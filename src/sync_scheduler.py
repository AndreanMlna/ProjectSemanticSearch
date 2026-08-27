"""
src/sync_scheduler.py
=====================
Layanan Scheduler / Daemon Background untuk memeriksa dan menyinkronkan dataset
metadata SERANAH secara otomatis setiap 2 jam sekali.

Fitur:
1. Memeriksa kesesuaian dataset secara berkala (default: setiap 2 jam).
2. Hanya melakukan unduh ulang, preprocessing, dan re-indexing ChromaDB
   apabila terdapat perbedaan data (jumlah dokumen atau dokumen baru).
3. Memberikan log waktu pelaksanaan berikutnya secara informatif.
4. Mendukung graceful shutdown (Ctrl+C atau sinyal terminate).
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Muat variabel environment
load_dotenv()

# Import modul sinkronisasi inti
from src.sync_seranah_archives import check_and_sync

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SyncScheduler")

# Interval sinkronisasi (default: 2 jam = 7200 detik)
SYNC_INTERVAL_HOURS = float(os.getenv("SYNC_INTERVAL_HOURS", "2"))
INTERVAL_SECONDS = int(SYNC_INTERVAL_HOURS * 3600)

_keep_running = True


def handle_shutdown(signum, frame):
    """Menangani permintaan shutdown secara aman (Ctrl+C)."""
    global _keep_running
    logger.info("\n[*] Menerima sinyal penghentian scheduler. Bersiap mematikan layanan...")
    _keep_running = False


def start_scheduler():
    """
    Menjalankan scheduler loop berkala untuk sinkronisasi dataset.
    """
    global _keep_running

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    print("=" * 70)
    print("      LAYANAN SCHEDULER SINKRONISASI DATASET SERANAH OTOMATIS        ")
    print("=" * 70)
    logger.info(f"[*] Interval Pengecekan: Setiap {SYNC_INTERVAL_HOURS} jam ({INTERVAL_SECONDS} detik).")
    logger.info("[*] Memulai siklus pengecekan pertama sekarang...\n")

    iteration = 1

    while _keep_running:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"=== [Iterasi #{iteration}] Menjalankan Pengecekan Dataset ({now_str}) ===")

        try:
            # Jalankan pengecekan dan sinkronisasi
            result = check_and_sync(auto_reindex=True)
            status = result.get("status")
            updated = result.get("updated")

            if updated:
                logger.info(f"[⚡] Hasil: Dataset dan Database Vektor BERHASIL DIPERBARUI ke {result.get('local_count')} dokumen.")
            else:
                logger.info(f"[✅] Hasil: Dataset SUDAH SINKRON ({result.get('local_count')} dokumen). Tidak perlu re-indexing.")

        except Exception as e:
            logger.error(f"[!] Terjadi kesalahan pada iterasi #{iteration}: {e}", exc_info=True)

        if not _keep_running:
            break

        # Hitung waktu jadwal berikutnya
        next_run = datetime.now() + timedelta(seconds=INTERVAL_SECONDS)
        next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n[*] Pengecekan berikutnya dijadwalkan pada: {next_run_str}")
        logger.info(f"[*] Menunggu selama {SYNC_INTERVAL_HOURS} jam...\n")

        # Tidur per detik agar responsif terhadap sinyal Ctrl+C
        sleep_elapsed = 0
        while _keep_running and sleep_elapsed < INTERVAL_SECONDS:
            time.sleep(1)
            sleep_elapsed += 1

        iteration += 1

    logger.info("[+] Layanan Sync Scheduler telah berhenti dengan aman. Sampai jumpa!")


if __name__ == "__main__":
    start_scheduler()
