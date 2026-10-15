"""Canonical scalar rules reused by domain code and public schemas."""

NAME_LIMIT = 120
PURPOSE_LIMIT = 2000
DESCRIPTION_LIMIT = 20000


def text(value: object, field: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}: потрібен текст.")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{field}: введіть значення.")
    if len(value) > limit:
        raise ValueError(f"{field}: максимум {limit} символів.")
    return value
