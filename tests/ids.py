"""Fixed, readable UUIDs for tests (the backend's TestIds.uid): uid(7) is 00000000-0000-0000-0000-000000000007."""
from uuid import UUID


def uid(n: int) -> str:
    return str(UUID(int=n))
