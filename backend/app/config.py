import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API 설정
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Bias-Free Synthetic Data Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    WORKERS: int = int(os.getenv("WORKERS", 4))
    
    # 데이터베이스 설정
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://user:password@localhost/synthdata"
    )
    
    # Redis 설정
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # 보안 설정
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML 모델 설정
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./models_cache")
    MAX_ENSEMBLE_GENERATORS: int = 10
    MIN_ENSEMBLE_GENERATORS: int = 3
    
    # 생성 제약사항
    MAX_TEXT_GENERATION: int = 100000  # 최대 텍스트 생성 수
    MAX_IMAGE_GENERATION: int = 10000  # 최대 이미지 생성 수
    GENERATION_TIMEOUT: int = 3600  # 1시간 (초)
    
    # 품질 메트릭 임계값
    MIN_QUALITY_SCORE: float = 0.85  # 최소 품질 점수 (85%)
    MAX_BIAS_SCORE: float = 0.25  # 최대 편향성 점수 (25%)
    MIN_SEMANTIC_SIMILARITY: float = 0.70  # 최소 의미적 유사성
    
    # 개인정보보호 설정
    K_ANONYMITY: int = 5  # k-익명성 파라미터
    EPSILON: float = 1.0  # 차등 개인정보보호 파라미터
    MAX_REIDENTIFICATION_RISK: float = 0.05  # 최대 재식별 위험도
    
    # 성능 설정
    BATCH_SIZE: int = 32
    MAX_CONCURRENT_REQUESTS: int = 100
    CACHE_TTL: int = 3600  # 캐시 TTL (초)
    
    # 모델 설정
    TEXT_MODELS: List[str] = [
        "gpt2",
        "gpt2-medium", 
        "distilbert-base-uncased",
        "bert-base-uncased",
        "t5-small"
    ]
    
    # 도메인별 설정
    DOMAINS: Dict[str, Dict[str, Any]] = {
        "medical": {
            "privacy_level": "high",
            "min_quality": 0.95,
            "compliance": ["HIPAA"],
            "constraints": ["medical_protocol", "patient_privacy"]
        },
        "financial": {
            "privacy_level": "high", 
            "min_quality": 0.88,
            "compliance": ["PCI-DSS", "SOX"],
            "constraints": ["transaction_validity", "regulatory_rules"]
        },
        "general": {
            "privacy_level": "medium",
            "min_quality": 0.85,
            "compliance": ["GDPR"],
            "constraints": ["logical_consistency"]
        }
    }
    
    # 편향성 메트릭 설정
    BIAS_METRICS: List[str] = [
        "demographic_parity",
        "equalized_odds",
        "average_odds_difference",
        "statistical_parity_difference",
        "disparate_impact_ratio"
    ]
    
    # 품질 메트릭 설정  
    QUALITY_METRICS: List[str] = [
        "column_distribution_stability",
        "deep_structure_stability",
        "column_correlation_stability",
        "text_structure_similarity",
        "semantic_similarity"
    ]
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()