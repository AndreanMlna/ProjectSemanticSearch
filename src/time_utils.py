from datetime import datetime, timedelta, timezone

JAKARTA_TZ = timezone(timedelta(hours=7))


def now_jakarta() -> datetime:
    return datetime.now(JAKARTA_TZ)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)