from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> bool:
    return True