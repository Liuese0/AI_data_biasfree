from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

# Enum 정의
class DataType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABULAR = "tabular"

class Domain(str, Enum):
    MEDICAL = "medical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    GENERAL = "general"

class OutputFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    TXT = "txt"

class GenerationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"

# 생성 요청 스키마
class GenerationRequest(BaseModel):
    """데이터 생성 요청 스키마"""
    prompt: str = Field(..., description="자연어 데이터셋 요청 프롬프트")
    data_type: DataType = Field(DataType.TEXT, description="생성할 데이터 타입")
    domain: Domain = Field(Domain.GENERAL, description="도메인 선택")
    quantity: int = Field(100, ge=1, le=100000, description="생성할 데이터 수량")
    output_format: OutputFormat = Field(OutputFormat.JSON, description="출력 형식")
    
    # 편향성 제어 파라미터
    bias_mitigation_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "demographic_balance": True,
            "gender_balance": True,
            "age_balance": True,
            "cultural_diversity": True,
            "target_bias_reduction": 0.15  # 15% 편향성 감소 목표
        }
    )
    
    # 품질 제약사항
    quality_constraints: Dict[str, float] = Field(
        default_factory=lambda: {
            "min_quality_score": 0.85,
            "min_semantic_similarity": 0.70,
            "max_reidentification_risk": 0.05
        }
    )
    
    # 도메인별 제약사항
    domain_constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="도메인 특화 제약사항"
    )
    
    # 앙상블 설정
    ensemble_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "num_generators": 5,
            "voting_method": "weighted_average",
            "diversity_weight": 0.3
        }
    )
    
    @validator('quantity')
    def validate_quantity(cls, v, values):
        """데이터 타입별 수량 제한 검증"""
        if 'data_type' in values:
            if values['data_type'] == DataType.TEXT and v > 100000:
                raise ValueError("텍스트 데이터는 최대 100,000개까지 생성 가능합니다")
            elif values['data_type'] == DataType.IMAGE and v > 10000:
                raise ValueError("이미지 데이터는 최대 10,000개까지 생성 가능합니다")
        return v

class GenerationResponse(BaseModel):
    """생성 응답 스키마"""
    job_id: str = Field(..., description="생성 작업 ID")
    status: GenerationStatus = Field(..., description="작업 상태")
    created_at: datetime = Field(..., description="생성 시작 시간")
    estimated_completion: Optional[datetime] = Field(None, description="예상 완료 시간")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="진행률")
    message: str = Field("", description="상태 메시지")

# 검증 관련 스키마
class ValidationRequest(BaseModel):
    """데이터 검증 요청 스키마"""
    job_id: str = Field(..., description="검증할 생성 작업 ID")
    validation_type: Literal["quality", "bias", "privacy", "all"] = Field(
        "all", 
        description="검증 유형"
    )
    reference_data: Optional[Dict[str, Any]] = Field(
        None,
        description="비교용 참조 데이터"
    )

class ValidationResult(BaseModel):
    """검증 결과 스키마"""
    job_id: str
    validation_timestamp: datetime
    
    # 품질 메트릭
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="품질 메트릭 결과"
    )
    
    # 편향성 메트릭
    bias_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="편향성 메트릭 결과"
    )
    
    # 개인정보보호 메트릭
    privacy_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="개인정보보호 메트릭"
    )
    
    # 종합 점수
    overall_quality_score: float = Field(0.0, ge=0.0, le=1.0)
    overall_bias_score: float = Field(0.0, ge=0.0, le=1.0)
    privacy_compliance: bool = Field(False)
    
    # 권장사항
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# 메트릭 관련 스키마
class MetricsRequest(BaseModel):
    """메트릭 조회 요청"""
    job_ids: Optional[List[str]] = Field(None, description="조회할 작업 ID 목록")
    start_date: Optional[datetime] = Field(None, description="시작 날짜")
    end_date: Optional[datetime] = Field(None, description="종료 날짜")
    metric_types: List[str] = Field(
        default_factory=lambda: ["quality", "bias", "performance"],
        description="조회할 메트릭 유형"
    )

class MetricsResponse(BaseModel):
    """메트릭 응답"""
    timestamp: datetime
    metrics: Dict[str, Any]
    aggregated_stats: Dict[str, float] = Field(
        default_factory=dict,
        description="집계된 통계"
    )
    trends: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="시간별 트렌드"
    )

# 샘플 데이터 스키마
class DataSample(BaseModel):
    """생성된 데이터 샘플"""
    sample_id: str
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float = Field(0.0, ge=0.0, le=1.0)
    bias_indicators: Dict[str, float] = Field(default_factory=dict)

class GeneratedDataset(BaseModel):
    """생성된 데이터셋"""
    job_id: str
    status: GenerationStatus
    data_type: DataType
    domain: Domain
    quantity: int
    actual_quantity: int = Field(..., description="실제 생성된 데이터 수")
    
    # 데이터 샘플
    samples: List[DataSample] = Field(default_factory=list)
    
    # 메타데이터
    generation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="생성 메타데이터"
    )
    
    # 검증 결과
    validation_result: Optional[ValidationResult] = None
    
    # 통계
    statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="데이터셋 통계"
    )
    
    created_at: datetime
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None

# 앙상블 관련 스키마
class GeneratorInfo(BaseModel):
    """생성기 정보"""
    generator_id: str
    model_name: str
    status: Literal["active", "inactive", "error"]
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    weight: float = Field(0.0, ge=0.0, le=1.0)
    specialization: Optional[str] = None

class EnsembleStatus(BaseModel):
    """앙상블 상태"""
    active_generators: List[GeneratorInfo]
    total_generators: int
    ensemble_performance: Dict[str, float]
    last_updated: datetime

# 에러 스키마
class ErrorResponse(BaseModel):
    """에러 응답"""
    error_code: str
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# 작업 상태 스키마
class JobStatus(BaseModel):
    """작업 상태 조회 응답"""
    job_id: str
    status: GenerationStatus
    progress: float
    current_step: str
    steps_completed: List[str]
    steps_remaining: List[str]
    estimated_time_remaining: Optional[int] = Field(None, description="남은 시간(초)")
    logs: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime