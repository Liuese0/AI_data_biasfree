from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy import select, update
import uuid
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Base 클래스 생성
Base = declarative_base()

# 데이터베이스 엔진 생성
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=40
)

# 세션 팩토리 생성
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 모델 정의
class GenerationJob(Base):
    """생성 작업 모델"""
    __tablename__ = "generation_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt = Column(Text, nullable=False)
    data_type = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    output_format = Column(String, nullable=False)
    
    # 설정
    bias_mitigation_config = Column(JSON, default={})
    quality_constraints = Column(JSON, default={})
    domain_constraints = Column(JSON, default={})
    ensemble_config = Column(JSON, default={})
    
    # 상태
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    message = Column(Text, default="")
    
    # 결과
    actual_quantity = Column(Integer, default=0)
    generation_metadata = Column(JSON, default={})
    statistics = Column(JSON, default={})
    download_url = Column(String, nullable=True)
    
    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    estimated_completion = Column(DateTime, nullable=True)
    
    # 관계
    validation_results = relationship("ValidationResult", back_populates="job")
    generated_samples = relationship("GeneratedSample", back_populates="job")
    metrics = relationship("GenerationMetric", back_populates="job")

class GeneratedSample(Base):
    """생성된 샘플 모델"""
    __tablename__ = "generated_samples"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("generation_jobs.id"))
    content = Column(JSON, nullable=False)
    metadata = Column(JSON, default={})
    quality_score = Column(Float, default=0.0)
    bias_indicators = Column(JSON, default={})
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 관계
    job = relationship("GenerationJob", back_populates="generated_samples")

class ValidationResult(Base):
    """검증 결과 모델"""
    __tablename__ = "validation_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("generation_jobs.id"))
    validation_type = Column(String, nullable=False)
    
    # 메트릭 결과
    quality_metrics = Column(JSON, default={})
    bias_metrics = Column(JSON, default={})
    privacy_metrics = Column(JSON, default={})
    
    # 종합 점수
    overall_quality_score = Column(Float, default=0.0)
    overall_bias_score = Column(Float, default=0.0)
    privacy_compliance = Column(Boolean, default=False)
    
    # 권장사항
    recommendations = Column(JSON, default=[])
    warnings = Column(JSON, default=[])
    
    validation_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # 관계
    job = relationship("GenerationJob", back_populates="validation_results")

class GenerationMetric(Base):
    """생성 메트릭 모델"""
    __tablename__ = "generation_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("generation_jobs.id"))
    metric_type = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metadata = Column(JSON, default={})
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # 관계
    job = relationship("GenerationJob", back_populates="metrics")

class GeneratorModel(Base):
    """생성기 모델 정보"""
    __tablename__ = "generator_models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, unique=True, nullable=False)
    model_type = Column(String, nullable=False)
    status = Column(String, default="inactive")
    
    # 성능 메트릭
    total_generations = Column(Integer, default=0)
    average_quality_score = Column(Float, default=0.0)
    average_bias_score = Column(Float, default=0.0)
    average_generation_time = Column(Float, default=0.0)
    
    # 가중치
    ensemble_weight = Column(Float, default=0.2)
    specialization = Column(String, nullable=True)
    
    # 설정
    configuration = Column(JSON, default={})
    
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 데이터베이스 초기화 함수
async def init_db():
    """데이터베이스 초기화"""
    try:
        async with engine.begin() as conn:
            # 테이블 생성
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# 데이터베이스 세션 의존성
async def get_db() -> AsyncSession:
    """데이터베이스 세션 제공"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# CRUD 유틸리티 함수들
class JobRepository:
    """생성 작업 레포지토리"""
    
    @staticmethod
    async def create_job(session: AsyncSession, job_data: Dict[str, Any]) -> GenerationJob:
        """새 작업 생성"""
        job = GenerationJob(**job_data)
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job
    
    @staticmethod
    async def get_job(session: AsyncSession, job_id: str) -> Optional[GenerationJob]:
        """작업 조회"""
        result = await session.execute(
            select(GenerationJob).where(GenerationJob.id == job_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_job_status(
        session: AsyncSession, 
        job_id: str, 
        status: str,
        progress: float = None,
        message: str = None
    ):
        """작업 상태 업데이트"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        if progress is not None:
            update_data["progress"] = progress
        if message is not None:
            update_data["message"] = message
            
        await session.execute(
            update(GenerationJob).where(GenerationJob.id == job_id).values(**update_data)
        )
        await session.commit()
    
    @staticmethod
    async def get_jobs_by_status(
        session: AsyncSession, 
        status: str,
        limit: int = 100
    ) -> List[GenerationJob]:
        """상태별 작업 조회"""
        result = await session.execute(
            select(GenerationJob)
            .where(GenerationJob.status == status)
            .limit(limit)
            .order_by(GenerationJob.created_at.desc())
        )
        return result.scalars().all()

class SampleRepository:
    """생성 샘플 레포지토리"""
    
    @staticmethod
    async def create_samples(
        session: AsyncSession, 
        job_id: str,
        samples: List[Dict[str, Any]]
    ):
        """샘플 일괄 생성"""
        sample_objects = [
            GeneratedSample(job_id=job_id, **sample) 
            for sample in samples
        ]
        session.add_all(sample_objects)
        await session.commit()
    
    @staticmethod
    async def get_samples(
        session: AsyncSession,
        job_id: str,
        limit: int = None
    ) -> List[GeneratedSample]:
        """작업별 샘플 조회"""
        query = select(GeneratedSample).where(GeneratedSample.job_id == job_id)
        if limit:
            query = query.limit(limit)
        result = await session.execute(query)
        return result.scalars().all()

class MetricRepository:
    """메트릭 레포지토리"""
    
    @staticmethod
    async def save_metric(
        session: AsyncSession,
        job_id: str,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        metadata: Dict[str, Any] = None
    ):
        """메트릭 저장"""
        metric = GenerationMetric(
            job_id=job_id,
            metric_type=metric_type,
            metric_name=metric_name,
            metric_value=metric_value,
            metadata=metadata or {}
        )
        session.add(metric)
        await session.commit()
    
    @staticmethod
    async def get_metrics(
        session: AsyncSession,
        job_id: str = None,
        metric_type: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[GenerationMetric]:
        """메트릭 조회"""
        query = select(GenerationMetric)
        
        if job_id:
            query = query.where(GenerationMetric.job_id == job_id)
        if metric_type:
            query = query.where(GenerationMetric.metric_type == metric_type)
        if start_date:
            query = query.where(GenerationMetric.timestamp >= start_date)
        if end_date:
            query = query.where(GenerationMetric.timestamp <= end_date)
            
        result = await session.execute(query.order_by(GenerationMetric.timestamp.desc()))
        return result.scalars().all()