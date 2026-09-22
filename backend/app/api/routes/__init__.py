"""routes — API endpoint routers."""

from backend.app.api.routes.assessment import router as assessment_router
from backend.app.api.routes.claims import router as claims_router
from backend.app.api.routes.dashboard import router as dashboard_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.images import router as images_router
from backend.app.api.routes.models import router as models_router
from backend.app.api.routes.reviews import router as reviews_router

__all__ = [
    "health_router",
    "claims_router",
    "images_router",
    "assessment_router",
    "reviews_router",
    "dashboard_router",
    "models_router",
]
