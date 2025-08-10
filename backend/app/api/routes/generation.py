from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import uuid
import logging

from app.models.schemas import (
    GenerationRequest,
    GenerationResponse,
    GeneratedDataset,
    JobStatus,
    DataSample
)
from app.models.database import get_db, JobRepository, SampleRepository, MetricRepository
from app.core.ensemble.ensemble_manager import EnsembleManager
from app.core.validation.quality_metrics import QualityMetrics
from app.core.validation.bias_metrics import BiasMetrics
from app.core.constraints.domain_rules import DomainConstraintEngine
from app.config import settings
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter()

# 전역 인스턴스
quality_metrics = QualityMetrics()
bias_metrics = BiasMetrics()
constraint_engine = DomainConstraintEngine()

# 진행 중인 작업 저장소
active_jobs: Dict[str, Dict[str, Any]] = {}

@router.post("/generate", response_model=GenerationResponse)
async def generate_data(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """합성 데이터 생성 요청"""
    
    try:
        # 요청 검증
        if request.quantity > settings.MAX_TEXT_GENERATION:
            raise HTTPException(
                status_code=400,
                detail=f"Quantity exceeds maximum limit of {settings.MAX_TEXT_GENERATION}"
            )
        
        # 작업 ID 생성
        job_id = str(uuid.uuid4())
        
        # 예상 완료 시간 계산
        estimated_time = calculate_estimated_time(
            request.quantity,
            request.data_type
        )
        estimated_completion = datetime.utcnow() + timedelta(seconds=estimated_time)
        
        # DB에 작업 저장
        job_data = {
            "id": job_id,
            "prompt": request.prompt,
            "data_type": request.data_type.value,
            "domain": request.domain.value,
            "quantity": request.quantity,
            "output_format": request.output_format.value,
            "bias_mitigation_config": request.bias_mitigation_config,
            "quality_constraints": request.quality_constraints,
            "domain_constraints": request.domain_constraints or {},
            "ensemble_config": request.ensemble_config,
            "status": "pending",
            "estimated_completion": estimated_completion
        }
        
        job = await JobRepository.create_job(db, job_data)
        
        # 활성 작업에 추가
        active_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "started_at": datetime.utcnow()
        }
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            process_generation,
            job_id,
            request
        )
        
        logger.info(f"Generation job created: {job_id}")
        
        return GenerationResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow(),
            estimated_completion=estimated_completion,
            progress=0.0,
            message="Generation job queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to create generation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """작업 상태 조회"""
    
    # DB에서 작업 조회
    job = await JobRepository.get_job(db, job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 활성 작업 정보 병합
    if job_id in active_jobs:
        active_info = active_jobs[job_id]
        progress = active_info.get("progress", 0.0)
        current_step = active_info.get("current_step", "initializing")
    else:
        progress = job.progress
        current_step = "completed" if job.status == "completed" else "processing"
    
    # 남은 시간 계산
    if job.estimated_completion and job.status not in ["completed", "failed"]:
        remaining_time = (job.estimated_completion - datetime.utcnow()).total_seconds()
        estimated_time_remaining = max(0, int(remaining_time))
    else:
        estimated_time_remaining = None
    
    # 단계 정보
    steps_completed = []
    steps_remaining = []
    
    if job.status == "completed":
        steps_completed = ["initialization", "generation", "validation", "storage"]
    elif job.status == "processing":
        steps_completed = ["initialization"]
        if progress > 0.3:
            steps_completed.append("generation")
        if progress > 0.7:
            steps_completed.append("validation")
        
        if "generation" not in steps_completed:
            steps_remaining.append("generation")
        if "validation" not in steps_completed:
            steps_remaining.append("validation")
        steps_remaining.append("storage")
    else:
        steps_remaining = ["initialization", "generation", "validation", "storage"]
    
    return JobStatus(
        job_id=job_id,
        status=job.status,
        progress=progress,
        current_step=current_step,
        steps_completed=steps_completed,
        steps_remaining=steps_remaining,
        estimated_time_remaining=estimated_time_remaining,
        logs=[job.message] if job.message else [],
        created_at=job.created_at,
        updated_at=job.updated_at
    )

@router.get("/result/{job_id}", response_model=GeneratedDataset)
async def get_generation_result(
    job_id: str,
    limit: int = Query(100, description="Maximum number of samples to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db)
):
    """생성 결과 조회"""
    
    # 작업 조회
    job = await JobRepository.get_job(db, job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in ["completed", "validated"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet. Current status: {job.status}"
        )
    
    # 샘플 조회
    samples = await SampleRepository.get_samples(db, job_id, limit=limit)
    
    # DataSample 객체로 변환
    data_samples = []
    for sample in samples[offset:offset+limit]:
        data_samples.append(DataSample(
            sample_id=sample.id,
            content=sample.content,
            metadata=sample.metadata or {},
            quality_score=sample.quality_score,
            bias_indicators=sample.bias_indicators or {}
        ))
    
    # 검증 결과 조회
    validation_result = None
    if job.validation_results:
        latest_validation = job.validation_results[-1]
        validation_result = {
            "quality_metrics": latest_validation.quality_metrics,
            "bias_metrics": latest_validation.bias_metrics,
            "privacy_metrics": latest_validation.privacy_metrics,
            "overall_quality_score": latest_validation.overall_quality_score,
            "overall_bias_score": latest_validation.overall_bias_score,
            "privacy_compliance": latest_validation.privacy_compliance
        }
    
    return GeneratedDataset(
        job_id=job_id,
        status=job.status,
        data_type=job.data_type,
        domain=job.domain,
        quantity=job.quantity,
        actual_quantity=job.actual_quantity or len(data_samples),
        samples=data_samples,
        generation_metadata=job.generation_metadata or {},
        validation_result=validation_result,
        statistics=job.statistics or {},
        created_at=job.created_at,
        completed_at=job.completed_at,
        download_url=job.download_url
    )

@router.post("/cancel/{job_id}")
async def cancel_generation(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """생성 작업 취소"""
    
    # 작업 조회
    job = await JobRepository.get_job(db, job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    # 상태 업데이트
    await JobRepository.update_job_status(
        db,
        job_id,
        "cancelled",
        message="Job cancelled by user"
    )
    
    # 활성 작업에서 제거
    if job_id in active_jobs:
        active_jobs[job_id]["cancelled"] = True
    
    return {"message": "Job cancelled successfully", "job_id": job_id}

@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of jobs to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db)
):
    """작업 목록 조회"""
    
    if status:
        jobs = await JobRepository.get_jobs_by_status(db, status, limit=limit)
    else:
        # 모든 작업 조회 (간단한 구현)
        jobs = []
        for job_status in ["pending", "processing", "completed", "failed"]:
            status_jobs = await JobRepository.get_jobs_by_status(
                db, 
                job_status, 
                limit=limit
            )
            jobs.extend(status_jobs)
    
    # 페이지네이션
    paginated_jobs = jobs[offset:offset+limit]
    
    # 응답 형식화
    job_list = []
    for job in paginated_jobs:
        job_list.append({
            "job_id": job.id,
            "prompt": job.prompt[:100] + "..." if len(job.prompt) > 100 else job.prompt,
            "data_type": job.data_type,
            "domain": job.domain,
            "quantity": job.quantity,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        })
    
    return {
        "total": len(jobs),
        "limit": limit,
        "offset": offset,
        "jobs": job_list
    }

@router.post("/regenerate/{job_id}")
async def regenerate_data(
    job_id: str,
    modifications: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """데이터 재생성"""
    
    # 원본 작업 조회
    original_job = await JobRepository.get_job(db, job_id)
    
    if not original_job:
        raise HTTPException(status_code=404, detail="Original job not found")
    
    # 새 작업 생성
    new_job_id = str(uuid.uuid4())
    
    # 수정사항 적용
    job_data = {
        "id": new_job_id,
        "prompt": modifications.get("prompt", original_job.prompt),
        "data_type": modifications.get("data_type", original_job.data_type),
        "domain": modifications.get("domain", original_job.domain),
        "quantity": modifications.get("quantity", original_job.quantity),
        "output_format": modifications.get("output_format", original_job.output_format),
        "bias_mitigation_config": modifications.get(
            "bias_mitigation_config",
            original_job.bias_mitigation_config
        ),
        "quality_constraints": modifications.get(
            "quality_constraints",
            original_job.quality_constraints
        ),
        "domain_constraints": modifications.get(
            "domain_constraints",
            original_job.domain_constraints
        ),
        "ensemble_config": modifications.get(
            "ensemble_config",
            original_job.ensemble_config
        ),
        "status": "pending"
    }
    
    new_job = await JobRepository.create_job(db, job_data)
    
    # 요청 객체 생성
    request = GenerationRequest(
        prompt=job_data["prompt"],
        data_type=job_data["data_type"],
        domain=job_data["domain"],
        quantity=job_data["quantity"],
        output_format=job_data["output_format"],
        bias_mitigation_config=job_data["bias_mitigation_config"],
        quality_constraints=job_data["quality_constraints"],
        domain_constraints=job_data["domain_constraints"],
        ensemble_config=job_data["ensemble_config"]
    )
    
    # 백그라운드 작업 시작
    background_tasks.add_task(
        process_generation,
        new_job_id,
        request
    )
    
    return {
        "message": "Regeneration started",
        "original_job_id": job_id,
        "new_job_id": new_job_id
    }

# 백그라운드 작업 함수
async def process_generation(job_id: str, request: GenerationRequest):
    """생성 작업 처리"""
    
    from app.main import ensemble_manager
    
    async with async_session() as db:
        try:
            logger.info(f"Starting generation for job {job_id}")
            
            # 상태 업데이트: processing
            await JobRepository.update_job_status(
                db,
                job_id,
                "processing",
                progress=0.1,
                message="Initializing generation"
            )
            
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "status": "processing",
                    "progress": 0.1,
                    "current_step": "initialization"
                })
            
            # 제약사항 준비
            constraints = {
                "domain": request.domain.value,
                "bias_mitigation": request.bias_mitigation_config,
                "quality_constraints": request.quality_constraints,
                "domain_constraints": request.domain_constraints,
                **request.ensemble_config
            }
            
            # 앙상블 생성 실행
            if job_id in active_jobs:
                active_jobs[job_id]["current_step"] = "generation"
                active_jobs[job_id]["progress"] = 0.3
            
            await JobRepository.update_job_status(
                db,
                job_id,
                "processing",
                progress=0.3,
                message="Generating synthetic data"
            )
            
            generated_data, ensemble_metrics = await ensemble_manager.generate_ensemble(
                prompt=request.prompt,
                quantity=request.quantity,
                domain=request.domain.value,
                constraints=constraints
            )
            
            logger.info(f"Generated {len(generated_data)} samples for job {job_id}")
            
            # 도메인 제약사항 적용
            if job_id in active_jobs:
                active_jobs[job_id]["current_step"] = "constraints"
                active_jobs[job_id]["progress"] = 0.5
            
            await JobRepository.update_job_status(
                db,
                job_id,
                "processing",
                progress=0.5,
                message="Applying domain constraints"
            )
            
            for item in generated_data:
                content = item.get("content", "")
                constrained_content = constraint_engine.apply_constraints(
                    content,
                    request.domain.value,
                    request.domain_constraints
                )
                item["content"] = constrained_content
            
            # 품질 검증
            if job_id in active_jobs:
                active_jobs[job_id]["current_step"] = "validation"
                active_jobs[job_id]["progress"] = 0.7
            
            await JobRepository.update_job_status(
                db,
                job_id,
                "processing",
                progress=0.7,
                message="Validating quality metrics"
            )
            
            quality_scores = await quality_metrics.calculate_quality_score(
                generated_data,
                domain=request.domain.value
            )
            
            # 편향성 검증
            bias_scores = await bias_metrics.calculate_bias_metrics(
                generated_data
            )
            
            # 샘플 저장
            if job_id in active_jobs:
                active_jobs[job_id]["current_step"] = "storage"
                active_jobs[job_id]["progress"] = 0.9
            
            await JobRepository.update_job_status(
                db,
                job_id,
                "processing",
                progress=0.9,
                message="Storing generated data"
            )
            
            samples_to_save = []
            for i, item in enumerate(generated_data):
                sample_data = {
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "quality_score": item.get("quality_score", 0.0),
                    "bias_indicators": item.get("bias_indicators", {})
                }
                samples_to_save.append(sample_data)
            
            await SampleRepository.create_samples(db, job_id, samples_to_save)
            
            # 메트릭 저장
            await MetricRepository.save_metric(
                db,
                job_id,
                "quality",
                "overall_quality_score",
                quality_scores.get("overall_quality_score", 0.0),
                quality_scores
            )
            
            await MetricRepository.save_metric(
                db,
                job_id,
                "bias",
                "overall_bias_score",
                bias_scores.get("overall_bias_score", 0.0),
                bias_scores
            )
            
            # 통계 계산
            statistics = {
                "total_generated": len(generated_data),
                "quality_metrics": quality_scores,
                "bias_metrics": bias_scores,
                "ensemble_metrics": ensemble_metrics,
                "generation_time": (
                    datetime.utcnow() - active_jobs[job_id]["started_at"]
                ).total_seconds() if job_id in active_jobs else 0
            }
            
            # 작업 완료
            await JobRepository.update_job_status(
                db,
                job_id,
                "completed",
                progress=1.0,
                message="Generation completed successfully"
            )
            
            # 추가 필드 업데이트
            job = await JobRepository.get_job(db, job_id)
            job.actual_quantity = len(generated_data)
            job.statistics = statistics
            job.generation_metadata = ensemble_metrics
            job.completed_at = datetime.utcnow()
            await db.commit()
            
            # 활성 작업에서 제거
            if job_id in active_jobs:
                del active_jobs[job_id]
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Generation failed for job {job_id}: {e}")
            
            # 실패 상태 업데이트
            await JobRepository.update_job_status(
                db,
                job_id,
                "failed",
                message=f"Generation failed: {str(e)}"
            )
            
            # 활성 작업에서 제거
            if job_id in active_jobs:
                del active_jobs[job_id]
            
            raise

def calculate_estimated_time(quantity: int, data_type: str) -> int:
    """예상 소요 시간 계산 (초)"""
    
    # 기본 시간 (초)
    base_time = {
        "text": 0.5,
        "image": 2.0,
        "tabular": 1.0
    }
    
    time_per_item = base_time.get(data_type, 1.0)
    
    # 배치 처리 고려
    batch_size = settings.BATCH_SIZE
    num_batches = (quantity + batch_size - 1) // batch_size
    
    # 오버헤드 추가
    overhead = 10  # 초기화 및 검증 시간
    
    estimated_time = (num_batches * batch_size * time_per_item) + overhead
    
    # 최대 시간 제한
    return min(int(estimated_time), settings.GENERATION_TIMEOUT)