import re

from fastapi import HTTPException, Request

from src.logging_utils import setup_logging
from src.rate_limiter import get_endpoint_rate_limiter

logger = setup_logging("helpers")


def extract_keywords(text: str) -> str:
    if not text:
        return "-"
    parts = re.split(r"kata\s+kunci\s*:?", text, flags=re.IGNORECASE)
    if len(parts) > 1 and parts[-1].strip():
        return parts[-1].strip()
    return "-"


def build_file_url(request: Request, filename: str) -> str:
    if not filename or filename == "-":
        return ""
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/files/{filename}"


def check_rate_limit(request: Request, endpoint: str) -> None:
    try:
        client_ip = request.client.host if request.client else "unknown"
        rate_limiter = get_endpoint_rate_limiter()
        allowed, info = rate_limiter.check_limit(endpoint=endpoint, client_id=client_ip)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded: endpoint=/{endpoint}, ip={client_ip}, used={info.get('requests_used')}/{info.get('requests_limit')}"
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too Many Requests",
                    "message": f"Batas request tercapai untuk endpoint /{endpoint}. Coba lagi dalam 1 menit.",
                    "limit": info.get("requests_limit"),
                    "reset_time": info.get("reset_time"),
                },
            )

        logger.debug(
            f"Rate limit OK: endpoint=/{endpoint}, ip={client_ip}, remaining={info.get('remaining')}"
        )
    except HTTPException:
        raise
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.warning(f"Rate limiter check failed (non-fatal): {e!s}")
