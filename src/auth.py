import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.config import API_SECRET_KEY, PUBLIC_ENDPOINTS
from src.logging_utils import setup_logging

logger = setup_logging("auth")

api_key_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="API Key Header"
)
http_bearer = HTTPBearer(auto_error=False, description="Bearer Token")


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(http_bearer),  # noqa: B008
) -> bool:
    path = request.url.path

    if path in PUBLIC_ENDPOINTS or path.startswith(
        ("/docs", "/redoc", "/openapi.json")
    ):
        return True

    token = api_key or (bearer.credentials if bearer else None)

    if not token or not secrets.compare_digest(token, API_SECRET_KEY):
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(
            f"Akses tidak sah ditolak: endpoint={path}, ip={client_ip}, token_provided={'yes' if token else 'no'}"
        )
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "error": "Unauthorized",
                "message": "Autentikasi gagal. Sediakan API Key yang valid melalui header 'X-API-Key' atau 'Authorization: Bearer <API_KEY>'.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
