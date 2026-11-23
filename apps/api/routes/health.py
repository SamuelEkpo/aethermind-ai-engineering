"""
Health and readiness endpoints.

/health      — liveness probe: is the process up? No external dependencies checked.
/health/ready — readiness probe: are downstream dependencies (DB, cache) reachable?
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness() -> dict[str, str]:
    # TODO(Phase 8): add real DB/Redis connectivity checks once those
    # clients are wired up in apps/api/core/.
    return {"status": "ready"}
