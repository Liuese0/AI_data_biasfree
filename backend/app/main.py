import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.config import settings
from app.api.routes import generation, validation, metrics
from app.models.database import init_db
from app.core.ensemble.ensemble_manager import EnsembleManager

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# 전역 앙상블 매니저 인스턴스
ensemble_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global ensemble_manager
    
    # 시작 시
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    
    # 데이터베이스 초기화
    await init_db()
    logger.info("Database initialized")
    
    # 앙상블 매니저 초기화
    ensemble_manager = EnsembleManager()
    await ensemble_manager.initialize()
    logger.info("Ensemble manager initialized")
    
    yield
    
    # 종료 시
    logger.info("Shutting down application")
    if ensemble_manager:
        await ensemble_manager.cleanup()
    logger.info("Cleanup completed")

# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 예외 처리
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "error": str(exc) if settings.DEBUG else "Internal server error"
        }
    )

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """시스템 헬스체크"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "ensemble_status": "active" if ensemble_manager and ensemble_manager.is_ready else "initializing"
    }

# 메트릭스 엔드포인트
@app.get("/metrics")
async def get_system_metrics():
    """시스템 메트릭스 조회"""
    if not ensemble_manager:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return {
        "active_generators": ensemble_manager.active_generators_count,
        "total_generations": ensemble_manager.total_generations,
        "average_quality_score": ensemble_manager.average_quality_score,
        "average_bias_score": ensemble_manager.average_bias_score,
    }

# API 라우터 등록
app.include_router(
    generation.router,
    prefix=f"{settings.API_V1_STR}/generation",
    tags=["generation"]
)

app.include_router(
    validation.router,
    prefix=f"{settings.API_V1_STR}/validation",
    tags=["validation"]
)

app.include_router(
    metrics.router,
    prefix=f"{settings.API_V1_STR}/metrics",
    tags=["metrics"]
)

# 루트 엔드포인트
@app.get("/")
async def root():
    """API 정보 반환"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "편향성 제거 합성 데이터 생성 플랫폼",
        "api_docs": f"{settings.API_V1_STR}/docs",
        "health_check": "/health",
        "features": {
            "text_generation": True,
            "image_generation": False,  # Phase 2에서 구현
            "ensemble_generation": True,
            "bias_mitigation": True,
            "quality_validation": True,
            "privacy_preservation": True
        },
        "supported_domains": list(settings.DOMAINS.keys()),
        "max_text_generation": settings.MAX_TEXT_GENERATION,
        "min_quality_score": settings.MIN_QUALITY_SCORE,
        "max_bias_score": settings.MAX_BIAS_SCORE
    }

# 앱 정보 엔드포인트
@app.get("/info")
async def get_info():
    """상세 시스템 정보 반환"""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "configuration": {
            "max_ensemble_generators": settings.MAX_ENSEMBLE_GENERATORS,
            "min_ensemble_generators": settings.MIN_ENSEMBLE_GENERATORS,
            "k_anonymity": settings.K_ANONYMITY,
            "epsilon": settings.EPSILON,
            "batch_size": settings.BATCH_SIZE,
            "supported_models": settings.TEXT_MODELS,
            "bias_metrics": settings.BIAS_METRICS,
            "quality_metrics": settings.QUALITY_METRICS
        },
        "domains": settings.DOMAINS,
        "limits": {
            "max_text_generation": settings.MAX_TEXT_GENERATION,
            "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS,
            "generation_timeout": settings.GENERATION_TIMEOUT
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )