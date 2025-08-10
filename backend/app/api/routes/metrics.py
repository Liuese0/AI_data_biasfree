from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging

from app.models.schemas import MetricsRequest, MetricsResponse
from app.models.database import (
    get_db,
    GenerationJob,
    GenerationMetric,
    ValidationResult,
    GeneratorModel,
    MetricRepository
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/collect", response_model=MetricsResponse)
async def collect_metrics(
    request: MetricsRequest,
    db: AsyncSession = Depends(get_db)
):
    """메트릭 수집 및 집계"""
    
    try:
        # 기간 설정
        start_date = request.start_date or datetime.utcnow() - timedelta(days=7)
        end_date = request.end_date or datetime.utcnow()
        
        # 메트릭 조회
        metrics_data = {}
        
        if "quality" in request.metric_types:
            quality_metrics = await collect_quality_metrics(
                db, request.job_ids, start_date, end_date
            )
            metrics_data["quality"] = quality_metrics
        
        if "bias" in request.metric_types:
            bias_metrics = await collect_bias_metrics(
                db, request.job_ids, start_date, end_date
            )
            metrics_data["bias"] = bias_metrics
        
        if "performance" in request.metric_types:
            performance_metrics = await collect_performance_metrics(
                db, request.job_ids, start_date, end_date
            )
            metrics_data["performance"] = performance_metrics
        
        # 집계 통계
        aggregated_stats = calculate_aggregated_stats(metrics_data)
        
        # 트렌드 분석
        trends = await analyze_trends(
            db, request.metric_types, start_date, end_date
        )
        
        return MetricsResponse(
            timestamp=datetime.utcnow(),
            metrics=metrics_data,
            aggregated_stats=aggregated_stats,
            trends=trends
        )
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_metrics_summary(
    period: str = Query("7d", description="Period: 1d, 7d, 30d, 90d"),
    db: AsyncSession = Depends(get_db)
):
    """메트릭 요약 조회"""
    
    # 기간 파싱
    period_map = {
        "1d": timedelta(days=1),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90)
    }
    
    delta = period_map.get(period, timedelta(days=7))
    start_date = datetime.utcnow() - delta
    
    # 작업 통계
    job_stats = await get_job_statistics(db, start_date)
    
    # 품질 통계
    quality_stats = await get_quality_statistics(db, start_date)
    
    # 편향성 통계
    bias_stats = await get_bias_statistics(db, start_date)
    
    # 생성기 성능
    generator_stats = await get_generator_statistics(db)
    
    return {
        "period": period,
        "start_date": start_date.isoformat(),
        "end_date": datetime.utcnow().isoformat(),
        "job_statistics": job_stats,
        "quality_statistics": quality_stats,
        "bias_statistics": bias_stats,
        "generator_performance": generator_stats
    }

@router.get("/realtime")
async def get_realtime_metrics():
    """실시간 메트릭 조회"""
    
    from app.main import ensemble_manager
    from app.api.routes.generation import active_jobs
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system_status": {
            "ensemble_ready": ensemble_manager.is_ready if ensemble_manager else False,
            "active_generators": ensemble_manager.active_generators_count if ensemble_manager else 0,
            "total_generations": ensemble_manager.total_generations if ensemble_manager else 0
        },
        "active_jobs": {
            "count": len(active_jobs),
            "jobs": [
                {
                    "job_id": job_id,
                    "status": info.get("status"),
                    "progress": info.get("progress", 0),
                    "current_step": info.get("current_step")
                }
                for job_id, info in active_jobs.items()
            ]
        },
        "current_performance": {
            "average_quality_score": ensemble_manager.average_quality_score if ensemble_manager else 0,
            "average_bias_score": ensemble_manager.average_bias_score if ensemble_manager else 0
        }
    }

@router.get("/trends/{metric_type}")
async def get_metric_trends(
    metric_type: str,
    granularity: str = Query("hour", description="hour, day, week, month"),
    period: int = Query(24, description="Number of periods to retrieve"),
    db: AsyncSession = Depends(get_db)
):
    """메트릭 트렌드 조회"""
    
    valid_metric_types = ["quality", "bias", "performance", "generation_volume"]
    if metric_type not in valid_metric_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric type. Must be one of: {valid_metric_types}"
        )
    
    # 시간 단위 설정
    time_units = {
        "hour": timedelta(hours=1),
        "day": timedelta(days=1),
        "week": timedelta(weeks=1),
        "month": timedelta(days=30)
    }
    
    unit = time_units.get(granularity, timedelta(hours=1))
    start_date = datetime.utcnow() - (unit * period)
    
    # 트렌드 데이터 조회
    trend_data = await get_trend_data(
        db, metric_type, start_date, granularity
    )
    
    return {
        "metric_type": metric_type,
        "granularity": granularity,
        "period": period,
        "start_date": start_date.isoformat(),
        "end_date": datetime.utcnow().isoformat(),
        "data_points": trend_data
    }

@router.get("/comparison")
async def compare_metrics(
    job_ids: List[str] = Query(..., description="Job IDs to compare"),
    db: AsyncSession = Depends(get_db)
):
    """작업 간 메트릭 비교"""
    
    if len(job_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 job IDs required for comparison"
        )
    
    comparison_data = []
    
    for job_id in job_ids[:10]:  # 최대 10개 비교
        # 작업 정보 조회
        result = await db.execute(
            select(GenerationJob).where(GenerationJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        
        if not job:
            comparison_data.append({
                "job_id": job_id,
                "error": "Job not found"
            })
            continue
        
        # 검증 결과 조회
        result = await db.execute(
            select(ValidationResult)
            .where(ValidationResult.job_id == job_id)
            .order_by(ValidationResult.validation_timestamp.desc())
            .limit(1)
        )
        validation = result.scalar_one_or_none()
        
        job_metrics = {
            "job_id": job_id,
            "domain": job.domain,
            "quantity": job.quantity,
            "status": job.status,
            "created_at": job.created_at.isoformat()
        }
        
        if validation:
            job_metrics.update({
                "quality_score": validation.overall_quality_score,
                "bias_score": validation.overall_bias_score,
                "privacy_compliant": validation.privacy_compliance
            })
        
        comparison_data.append(job_metrics)
    
    # 비교 통계
    comparison_stats = calculate_comparison_stats(comparison_data)
    
    return {
        "job_count": len(job_ids),
        "comparison_data": comparison_data,
        "statistics": comparison_stats
    }

@router.get("/benchmarks")
async def get_benchmarks(
    domain: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """벤치마크 메트릭 조회"""
    
    # 도메인별 벤치마크
    benchmarks = {
        "medical": {
            "target_quality": 0.95,
            "max_bias": 0.15,
            "privacy_compliance": 100,
            "generation_speed": 100  # samples per minute
        },
        "financial": {
            "target_quality": 0.88,
            "max_bias": 0.20,
            "privacy_compliance": 100,
            "generation_speed": 150
        },
        "legal": {
            "target_quality": 0.90,
            "max_bias": 0.18,
            "privacy_compliance": 100,
            "generation_speed": 120
        },
        "general": {
            "target_quality": 0.85,
            "max_bias": 0.25,
            "privacy_compliance": 95,
            "generation_speed": 200
        }
    }
    
    if domain:
        domain_benchmark = benchmarks.get(domain)
        if not domain_benchmark:
            raise HTTPException(status_code=404, detail="Domain not found")
        
        # 현재 성능과 비교
        current_performance = await get_current_performance(db, domain)
        
        return {
            "domain": domain,
            "benchmark": domain_benchmark,
            "current_performance": current_performance,
            "comparison": compare_to_benchmark(current_performance, domain_benchmark)
        }
    
    return {"benchmarks": benchmarks}

@router.get("/export")
async def export_metrics(
    format: str = Query("json", description="Export format: json, csv"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db)
):
    """메트릭 내보내기"""
    
    start_date = start_date or datetime.utcnow() - timedelta(days=30)
    end_date = end_date or datetime.utcnow()
    
    # 메트릭 데이터 조회
    metrics = await MetricRepository.get_metrics(
        db,
        start_date=start_date,
        end_date=end_date
    )
    
    if format == "csv":
        # CSV 형식으로 변환
        data_rows = []
        for metric in metrics:
            data_rows.append({
                "timestamp": metric.timestamp.isoformat(),
                "job_id": metric.job_id,
                "metric_type": metric.metric_type,
                "metric_name": metric.metric_name,
                "metric_value": metric.metric_value
            })
        
        df = pd.DataFrame(data_rows)
        csv_content = df.to_csv(index=False)
        
        from fastapi.responses import Response
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=metrics.csv"}
        )
    
    # JSON 형식
    return {
        "export_date": datetime.utcnow().isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_metrics": len(metrics),
        "metrics": [
            {
                "timestamp": m.timestamp.isoformat(),
                "job_id": m.job_id,
                "type": m.metric_type,
                "name": m.metric_name,
                "value": m.metric_value,
                "metadata": m.metadata
            }
            for m in metrics
        ]
    }

# 유틸리티 함수들
async def collect_quality_metrics(
    db: AsyncSession,
    job_ids: Optional[List[str]],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """품질 메트릭 수집"""
    
    query = select(GenerationMetric).where(
        GenerationMetric.metric_type == "quality",
        GenerationMetric.timestamp >= start_date,
        GenerationMetric.timestamp <= end_date
    )
    
    if job_ids:
        query = query.where(GenerationMetric.job_id.in_(job_ids))
    
    result = await db.execute(query)
    metrics = result.scalars().all()
    
    if not metrics:
        return {"count": 0, "average": 0}
    
    values = [m.metric_value for m in metrics]
    
    return {
        "count": len(metrics),
        "average": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "percentiles": {
            "25": np.percentile(values, 25),
            "50": np.percentile(values, 50),
            "75": np.percentile(values, 75),
            "90": np.percentile(values, 90)
        }
    }

async def collect_bias_metrics(
    db: AsyncSession,
    job_ids: Optional[List[str]],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """편향성 메트릭 수집"""
    
    query = select(GenerationMetric).where(
        GenerationMetric.metric_type == "bias",
        GenerationMetric.timestamp >= start_date,
        GenerationMetric.timestamp <= end_date
    )
    
    if job_ids:
        query = query.where(GenerationMetric.job_id.in_(job_ids))
    
    result = await db.execute(query)
    metrics = result.scalars().all()
    
    if not metrics:
        return {"count": 0, "average": 0}
    
    values = [m.metric_value for m in metrics]
    
    # 편향성 카테고리별 분석
    category_bias = {}
    for metric in metrics:
        metadata = metric.metadata or {}
        if "category_biases" in metadata:
            for category, bias_data in metadata["category_biases"].items():
                if category not in category_bias:
                    category_bias[category] = []
                category_bias[category].append(bias_data.get("overall", 0))
    
    category_averages = {
        cat: np.mean(values) for cat, values in category_bias.items()
    }
    
    return {
        "count": len(metrics),
        "average": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "category_averages": category_averages,
        "high_bias_count": sum(1 for v in values if v > settings.MAX_BIAS_SCORE)
    }

async def collect_performance_metrics(
    db: AsyncSession,
    job_ids: Optional[List[str]],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """성능 메트릭 수집"""
    
    # 작업 완료 시간 분석
    query = select(GenerationJob).where(
        GenerationJob.created_at >= start_date,
        GenerationJob.created_at <= end_date,
        GenerationJob.status == "completed"
    )
    
    if job_ids:
        query = query.where(GenerationJob.id.in_(job_ids))
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    if not jobs:
        return {"count": 0, "average_time": 0}
    
    # 처리 시간 계산
    processing_times = []
    generation_rates = []
    
    for job in jobs:
        if job.completed_at and job.created_at:
            processing_time = (job.completed_at - job.created_at).total_seconds()
            processing_times.append(processing_time)
            
            if job.actual_quantity and processing_time > 0:
                rate = job.actual_quantity / (processing_time / 60)  # samples per minute
                generation_rates.append(rate)
    
    return {
        "count": len(jobs),
        "total_samples_generated": sum(job.actual_quantity or 0 for job in jobs),
        "average_processing_time": np.mean(processing_times) if processing_times else 0,
        "median_processing_time": np.median(processing_times) if processing_times else 0,
        "average_generation_rate": np.mean(generation_rates) if generation_rates else 0,
        "success_rate": len(jobs) / len(jobs) if jobs else 0
    }

def calculate_aggregated_stats(metrics_data: Dict[str, Any]) -> Dict[str, float]:
    """집계 통계 계산"""
    
    stats = {}
    
    if "quality" in metrics_data:
        stats["average_quality"] = metrics_data["quality"].get("average", 0)
        stats["quality_std"] = metrics_data["quality"].get("std", 0)
    
    if "bias" in metrics_data:
        stats["average_bias"] = metrics_data["bias"].get("average", 0)
        stats["high_bias_ratio"] = (
            metrics_data["bias"].get("high_bias_count", 0) /
            max(1, metrics_data["bias"].get("count", 1))
        )
    
    if "performance" in metrics_data:
        stats["average_generation_rate"] = metrics_data["performance"].get(
            "average_generation_rate", 0
        )
        stats["success_rate"] = metrics_data["performance"].get("success_rate", 0)
    
    return stats

async def analyze_trends(
    db: AsyncSession,
    metric_types: List[str],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, List[float]]:
    """트렌드 분석"""
    
    trends = {}
    
    # 일별 트렌드 계산
    days = (end_date - start_date).days
    
    for metric_type in metric_types:
        daily_values = []
        
        for day in range(days + 1):
            day_start = start_date + timedelta(days=day)
            day_end = day_start + timedelta(days=1)
            
            query = select(func.avg(GenerationMetric.metric_value)).where(
                GenerationMetric.metric_type == metric_type,
                GenerationMetric.timestamp >= day_start,
                GenerationMetric.timestamp < day_end
            )
            
            result = await db.execute(query)
            avg_value = result.scalar()
            
            daily_values.append(avg_value if avg_value else 0)
        
        trends[metric_type] = daily_values
    
    return trends

async def get_job_statistics(
    db: AsyncSession,
    start_date: datetime
) -> Dict[str, Any]:
    """작업 통계 조회"""
    
    # 상태별 작업 수
    result = await db.execute(
        select(
            GenerationJob.status,
            func.count(GenerationJob.id)
        ).where(
            GenerationJob.created_at >= start_date
        ).group_by(GenerationJob.status)
    )
    
    status_counts = dict(result.all())
    
    # 도메인별 작업 수
    result = await db.execute(
        select(
            GenerationJob.domain,
            func.count(GenerationJob.id)
        ).where(
            GenerationJob.created_at >= start_date
        ).group_by(GenerationJob.domain)
    )
    
    domain_counts = dict(result.all())
    
    # 총 생성 샘플 수
    result = await db.execute(
        select(func.sum(GenerationJob.actual_quantity)).where(
            GenerationJob.created_at >= start_date,
            GenerationJob.status == "completed"
        )
    )
    
    total_samples = result.scalar() or 0
    
    return {
        "total_jobs": sum(status_counts.values()),
        "status_distribution": status_counts,
        "domain_distribution": domain_counts,
        "total_samples_generated": total_samples
    }

async def get_quality_statistics(
    db: AsyncSession,
    start_date: datetime
) -> Dict[str, Any]:
    """품질 통계 조회"""
    
    result = await db.execute(
        select(
            func.avg(ValidationResult.overall_quality_score),
            func.min(ValidationResult.overall_quality_score),
            func.max(ValidationResult.overall_quality_score),
            func.count(ValidationResult.id)
        ).where(
            ValidationResult.validation_timestamp >= start_date
        )
    )
    
    avg_quality, min_quality, max_quality, count = result.one()
    
    # 품질 기준 충족률
    result = await db.execute(
        select(func.count(ValidationResult.id)).where(
            ValidationResult.validation_timestamp >= start_date,
            ValidationResult.overall_quality_score >= settings.MIN_QUALITY_SCORE
        )
    )
    
    quality_pass_count = result.scalar() or 0
    
    return {
        "average_quality": avg_quality or 0,
        "min_quality": min_quality or 0,
        "max_quality": max_quality or 0,
        "total_validations": count or 0,
        "quality_pass_rate": quality_pass_count / max(1, count) if count else 0
    }

async def get_bias_statistics(
    db: AsyncSession,
    start_date: datetime
) -> Dict[str, Any]:
    """편향성 통계 조회"""
    
    result = await db.execute(
        select(
            func.avg(ValidationResult.overall_bias_score),
            func.min(ValidationResult.overall_bias_score),
            func.max(ValidationResult.overall_bias_score),
            func.count(ValidationResult.id)
        ).where(
            ValidationResult.validation_timestamp >= start_date
        )
    )
    
    avg_bias, min_bias, max_bias, count = result.one()
    
    # 편향성 기준 충족률
    result = await db.execute(
        select(func.count(ValidationResult.id)).where(
            ValidationResult.validation_timestamp >= start_date,
            ValidationResult.overall_bias_score <= settings.MAX_BIAS_SCORE
        )
    )
    
    bias_pass_count = result.scalar() or 0
    
    return {
        "average_bias": avg_bias or 0,
        "min_bias": min_bias or 0,
        "max_bias": max_bias or 0,
        "total_validations": count or 0,
        "bias_pass_rate": bias_pass_count / max(1, count) if count else 0
    }

async def get_generator_statistics(db: AsyncSession) -> Dict[str, Any]:
    """생성기 통계 조회"""
    
    result = await db.execute(
        select(GeneratorModel).where(
            GeneratorModel.status == "active"
        )
    )
    
    generators = result.scalars().all()
    
    generator_stats = []
    
    for gen in generators:
        generator_stats.append({
            "model_name": gen.model_name,
            "status": gen.status,
            "total_generations": gen.total_generations,
            "average_quality": gen.average_quality_score,
            "average_bias": gen.average_bias_score,
            "average_time": gen.average_generation_time,
            "weight": gen.ensemble_weight
        })
    
    return {
        "active_generators": len(generators),
        "generators": generator_stats
    }

async def get_trend_data(
    db: AsyncSession,
    metric_type: str,
    start_date: datetime,
    granularity: str
) -> List[Dict[str, Any]]:
    """트렌드 데이터 조회"""
    
    # 시간 단위별 그룹화
    if granularity == "hour":
        time_format = "%Y-%m-%d %H:00:00"
    elif granularity == "day":
        time_format = "%Y-%m-%d"
    elif granularity == "week":
        time_format = "%Y-%W"
    else:  # month
        time_format = "%Y-%m"
    
    trend_data = []
    
    if metric_type == "generation_volume":
        # 생성량 트렌드
        result = await db.execute(
            select(
                func.strftime(time_format, GenerationJob.created_at).label("period"),
                func.sum(GenerationJob.actual_quantity).label("value")
            ).where(
                GenerationJob.created_at >= start_date,
                GenerationJob.status == "completed"
            ).group_by("period")
        )
    else:
        # 메트릭 트렌드
        result = await db.execute(
            select(
                func.strftime(time_format, GenerationMetric.timestamp).label("period"),
                func.avg(GenerationMetric.metric_value).label("value")
            ).where(
                GenerationMetric.metric_type == metric_type,
                GenerationMetric.timestamp >= start_date
            ).group_by("period")
        )
    
    for row in result.all():
        trend_data.append({
            "period": row.period,
            "value": row.value or 0
        })
    
    return trend_data

def calculate_comparison_stats(comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """비교 통계 계산"""
    
    valid_data = [d for d in comparison_data if "error" not in d]
    
    if not valid_data:
        return {}
    
    quality_scores = [d.get("quality_score", 0) for d in valid_data]
    bias_scores = [d.get("bias_score", 0) for d in valid_data]
    
    return {
        "best_quality": {
            "job_id": max(valid_data, key=lambda x: x.get("quality_score", 0))["job_id"],
            "score": max(quality_scores) if quality_scores else 0
        },
        "lowest_bias": {
            "job_id": min(valid_data, key=lambda x: x.get("bias_score", 1))["job_id"],
            "score": min(bias_scores) if bias_scores else 0
        },
        "average_quality": np.mean(quality_scores) if quality_scores else 0,
        "average_bias": np.mean(bias_scores) if bias_scores else 0,
        "quality_variance": np.var(quality_scores) if quality_scores else 0,
        "bias_variance": np.var(bias_scores) if bias_scores else 0
    }

async def get_current_performance(
    db: AsyncSession,
    domain: str
) -> Dict[str, float]:
    """현재 성능 조회"""
    
    # 최근 30일 데이터
    start_date = datetime.utcnow() - timedelta(days=30)
    
    result = await db.execute(
        select(
            func.avg(ValidationResult.overall_quality_score),
            func.avg(ValidationResult.overall_bias_score),
            func.count(ValidationResult.id)
        ).join(
            GenerationJob
        ).where(
            GenerationJob.domain == domain,
            ValidationResult.validation_timestamp >= start_date
        )
    )
    
    avg_quality, avg_bias, count = result.one()
    
    # 개인정보보호 준수율
    result = await db.execute(
        select(func.count(ValidationResult.id)).join(
            GenerationJob
        ).where(
            GenerationJob.domain == domain,
            ValidationResult.validation_timestamp >= start_date,
            ValidationResult.privacy_compliance == True
        )
    )
    
    privacy_compliant_count = result.scalar() or 0
    
    # 생성 속도
    result = await db.execute(
        select(
            func.sum(GenerationJob.actual_quantity),
            func.sum(
                func.julianday(GenerationJob.completed_at) - 
                func.julianday(GenerationJob.created_at)
            )
        ).where(
            GenerationJob.domain == domain,
            GenerationJob.created_at >= start_date,
            GenerationJob.status == "completed"
        )
    )
    
    total_samples, total_time = result.one()
    
    generation_speed = 0
    if total_samples and total_time:
        generation_speed = total_samples / (total_time * 24 * 60)  # samples per minute
    
    return {
        "quality": avg_quality or 0,
        "bias": avg_bias or 0,
        "privacy_compliance": (privacy_compliant_count / max(1, count)) * 100 if count else 0,
        "generation_speed": generation_speed
    }

def compare_to_benchmark(
    current: Dict[str, float],
    benchmark: Dict[str, float]
) -> Dict[str, str]:
    """벤치마크와 비교"""
    
    comparison = {}
    
    if current["quality"] >= benchmark["target_quality"]:
        comparison["quality"] = "✅ Meets benchmark"
    else:
        diff = benchmark["target_quality"] - current["quality"]
        comparison["quality"] = f"❌ {diff:.2f} below benchmark"
    
    if current["bias"] <= benchmark["max_bias"]:
        comparison["bias"] = "✅ Meets benchmark"
    else:
        diff = current["bias"] - benchmark["max_bias"]
        comparison["bias"] = f"❌ {diff:.2f} above benchmark"
    
    if current["privacy_compliance"] >= benchmark["privacy_compliance"]:
        comparison["privacy"] = "✅ Meets benchmark"
    else:
        diff = benchmark["privacy_compliance"] - current["privacy_compliance"]
        comparison["privacy"] = f"❌ {diff:.1f}% below benchmark"
    
    if current["generation_speed"] >= benchmark["generation_speed"]:
        comparison["speed"] = "✅ Meets benchmark"
    else:
        diff = benchmark["generation_speed"] - current["generation_speed"]
        comparison["speed"] = f"❌ {diff:.1f} samples/min below benchmark"
    
    return comparison