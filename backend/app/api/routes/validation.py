from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import numpy as np

from app.models.schemas import (
    ValidationRequest,
    ValidationResult,
    GeneratedDataset
)
from app.models.database import (
    get_db,
    JobRepository,
    SampleRepository,
    ValidationResult as DBValidationResult
)
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

@router.post("/validate", response_model=ValidationResult)
async def validate_data(
    request: ValidationRequest,
    db: AsyncSession = Depends(get_db)
):
    """데이터 검증 실행"""
    
    try:
        # 작업 조회
        job = await JobRepository.get_job(db, request.job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status not in ["completed", "validated"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot validate job with status: {job.status}"
            )
        
        # 샘플 데이터 조회
        samples = await SampleRepository.get_samples(db, request.job_id)
        
        if not samples:
            raise HTTPException(
                status_code=404,
                detail="No samples found for validation"
            )
        
        # 검증 타입별 처리
        validation_results = {}
        
        if request.validation_type in ["quality", "all"]:
            quality_results = await validate_quality(
                samples,
                job.domain,
                request.reference_data
            )
            validation_results["quality_metrics"] = quality_results
        
        if request.validation_type in ["bias", "all"]:
            bias_results = await validate_bias(
                samples,
                job.domain
            )
            validation_results["bias_metrics"] = bias_results
        
        if request.validation_type in ["privacy", "all"]:
            privacy_results = await validate_privacy(
                samples,
                job.domain
            )
            validation_results["privacy_metrics"] = privacy_results
        
        # 종합 점수 계산
        overall_quality = calculate_overall_quality(validation_results)
        overall_bias = calculate_overall_bias(validation_results)
        privacy_compliant = check_privacy_compliance(validation_results)
        
        # 권장사항 생성
        recommendations = generate_recommendations(
            validation_results,
            overall_quality,
            overall_bias
        )
        
        # 경고사항 생성
        warnings = generate_warnings(
            validation_results,
            job.domain
        )
        
        # DB에 검증 결과 저장
        db_validation = DBValidationResult(
            job_id=request.job_id,
            validation_type=request.validation_type,
            quality_metrics=validation_results.get("quality_metrics", {}),
            bias_metrics=validation_results.get("bias_metrics", {}),
            privacy_metrics=validation_results.get("privacy_metrics", {}),
            overall_quality_score=overall_quality,
            overall_bias_score=overall_bias,
            privacy_compliance=privacy_compliant,
            recommendations=recommendations,
            warnings=warnings
        )
        
        db.add(db_validation)
        await db.commit()
        
        # 작업 상태 업데이트
        if job.status == "completed":
            await JobRepository.update_job_status(
                db,
                request.job_id,
                "validated",
                message="Validation completed"
            )
        
        logger.info(f"Validation completed for job {request.job_id}")
        
        return ValidationResult(
            job_id=request.job_id,
            validation_timestamp=datetime.utcnow(),
            quality_metrics=validation_results.get("quality_metrics", {}),
            bias_metrics=validation_results.get("bias_metrics", {}),
            privacy_metrics=validation_results.get("privacy_metrics", {}),
            overall_quality_score=overall_quality,
            overall_bias_score=overall_bias,
            privacy_compliance=privacy_compliant,
            recommendations=recommendations,
            warnings=warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/validate/{job_id}/history")
async def get_validation_history(
    job_id: str,
    limit: int = Query(10, description="Maximum number of validations to return"),
    db: AsyncSession = Depends(get_db)
):
    """검증 히스토리 조회"""
    
    # 작업 조회
    job = await JobRepository.get_job(db, job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 검증 결과 조회
    validation_history = []
    
    if job.validation_results:
        for validation in job.validation_results[-limit:]:
            validation_history.append({
                "validation_id": validation.id,
                "timestamp": validation.validation_timestamp.isoformat(),
                "type": validation.validation_type,
                "quality_score": validation.overall_quality_score,
                "bias_score": validation.overall_bias_score,
                "privacy_compliant": validation.privacy_compliance
            })
    
    return {
        "job_id": job_id,
        "total_validations": len(job.validation_results) if job.validation_results else 0,
        "history": validation_history
    }

@router.post("/validate/batch")
async def batch_validate(
    job_ids: List[str],
    validation_type: str = "all",
    db: AsyncSession = Depends(get_db)
):
    """배치 검증"""
    
    results = []
    
    for job_id in job_ids:
        try:
            request = ValidationRequest(
                job_id=job_id,
                validation_type=validation_type
            )
            
            result = await validate_data(request, db)
            results.append({
                "job_id": job_id,
                "status": "success",
                "result": result.dict()
            })
            
        except Exception as e:
            results.append({
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "total": len(job_ids),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }

@router.get("/quality-thresholds")
async def get_quality_thresholds():
    """품질 임계값 조회"""
    
    return {
        "thresholds": {
            "min_quality_score": settings.MIN_QUALITY_SCORE,
            "max_bias_score": settings.MAX_BIAS_SCORE,
            "min_semantic_similarity": settings.MIN_SEMANTIC_SIMILARITY,
            "max_reidentification_risk": settings.MAX_REIDENTIFICATION_RISK,
            "k_anonymity": settings.K_ANONYMITY,
            "epsilon": settings.EPSILON
        },
        "domains": {
            domain: {
                "min_quality": info.get("min_quality", 0.85),
                "privacy_level": info.get("privacy_level", "medium"),
                "compliance": info.get("compliance", [])
            }
            for domain, info in settings.DOMAINS.items()
        }
    }

@router.post("/validate/custom")
async def custom_validation(
    job_id: str,
    custom_metrics: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """커스텀 검증 메트릭 적용"""
    
    # 작업 조회
    job = await JobRepository.get_job(db, job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 샘플 데이터 조회
    samples = await SampleRepository.get_samples(db, job_id)
    
    if not samples:
        raise HTTPException(status_code=404, detail="No samples found")
    
    # 커스텀 메트릭 계산
    results = {}
    
    for metric_name, metric_config in custom_metrics.items():
        if metric_config.get("type") == "regex":
            pattern = metric_config.get("pattern")
            results[metric_name] = calculate_regex_metric(samples, pattern)
        
        elif metric_config.get("type") == "statistical":
            method = metric_config.get("method")
            results[metric_name] = calculate_statistical_metric(samples, method)
        
        elif metric_config.get("type") == "threshold":
            threshold = metric_config.get("threshold")
            field = metric_config.get("field")
            results[metric_name] = calculate_threshold_metric(samples, field, threshold)
    
    return {
        "job_id": job_id,
        "custom_metrics": results,
        "timestamp": datetime.utcnow().isoformat()
    }

# 유틸리티 함수들
async def validate_quality(
    samples: List[Any],
    domain: str,
    reference_data: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """품질 검증"""
    
    # 샘플 데이터를 딕셔너리 형식으로 변환
    sample_dicts = []
    for sample in samples:
        sample_dicts.append({
            "content": sample.content,
            "metadata": sample.metadata or {}
        })
    
    # 참조 데이터 처리
    ref_data = None
    if reference_data:
        ref_data = reference_data.get("samples", [])
    
    # 품질 메트릭 계산
    quality_scores = await quality_metrics.calculate_quality_score(
        sample_dicts,
        ref_data,
        domain
    )
    
    return quality_scores

async def validate_bias(
    samples: List[Any],
    domain: str
) -> Dict[str, Any]:
    """편향성 검증"""
    
    # 샘플 데이터 변환
    sample_dicts = []
    for sample in samples:
        sample_dicts.append({
            "content": sample.content,
            "metadata": sample.metadata or {}
        })
    
    # 편향성 메트릭 계산
    bias_scores = await bias_metrics.calculate_bias_metrics(
        sample_dicts,
        protected_attributes=["gender", "age", "ethnicity"]
    )
    
    return bias_scores

async def validate_privacy(
    samples: List[Any],
    domain: str
) -> Dict[str, Any]:
    """개인정보보호 검증"""
    
    privacy_metrics = {
        "k_anonymity_satisfied": True,
        "differential_privacy_satisfied": True,
        "reidentification_risk": 0.0,
        "sensitive_data_found": False,
        "privacy_violations": []
    }
    
    # 민감 정보 패턴
    sensitive_patterns = {
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    }
    
    violations = []
    sensitive_count = 0
    
    for sample in samples:
        content = str(sample.content)
        
        for pattern_name, pattern in sensitive_patterns.items():
            import re
            if re.search(pattern, content):
                sensitive_count += 1
                violations.append(f"Found {pattern_name} in sample {sample.id}")
                privacy_metrics["sensitive_data_found"] = True
    
    # 재식별 위험 계산
    total_samples = len(samples)
    if total_samples > 0:
        # 간단한 휴리스틱: 민감 정보가 포함된 샘플 비율
        privacy_metrics["reidentification_risk"] = sensitive_count / total_samples
    
    # k-익명성 체크 (간단한 버전)
    if total_samples < settings.K_ANONYMITY:
        privacy_metrics["k_anonymity_satisfied"] = False
        violations.append(f"Sample size {total_samples} < k={settings.K_ANONYMITY}")
    
    privacy_metrics["privacy_violations"] = violations
    
    return privacy_metrics

def calculate_overall_quality(validation_results: Dict[str, Any]) -> float:
    """종합 품질 점수 계산"""
    
    quality_metrics = validation_results.get("quality_metrics", {})
    
    if "overall_quality_score" in quality_metrics:
        return quality_metrics["overall_quality_score"]
    
    # 개별 메트릭 평균
    scores = []
    for key, value in quality_metrics.items():
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            scores.append(value)
    
    if scores:
        return np.mean(scores)
    
    return 0.5  # 기본값

def calculate_overall_bias(validation_results: Dict[str, Any]) -> float:
    """종합 편향성 점수 계산"""
    
    bias_metrics = validation_results.get("bias_metrics", {})
    
    if "overall_bias_score" in bias_metrics:
        return bias_metrics["overall_bias_score"]
    
    # 개별 메트릭 평균
    scores = []
    
    # 인구통계학적 패리티
    if "demographic_parity" in bias_metrics:
        dp = bias_metrics["demographic_parity"]
        if isinstance(dp, dict) and "overall" in dp:
            scores.append(1.0 - dp["overall"])  # 패리티가 높을수록 편향 낮음
    
    # 기타 편향성 점수
    for key in ["representation_bias", "stereotype_bias", "linguistic_bias"]:
        if key in bias_metrics:
            value = bias_metrics[key]
            if isinstance(value, (int, float)):
                scores.append(value)
    
    if scores:
        return np.mean(scores)
    
    return 0.5  # 기본값

def check_privacy_compliance(validation_results: Dict[str, Any]) -> bool:
    """개인정보보호 준수 확인"""
    
    privacy_metrics = validation_results.get("privacy_metrics", {})
    
    # 모든 조건 확인
    k_anonymity = privacy_metrics.get("k_anonymity_satisfied", False)
    differential_privacy = privacy_metrics.get("differential_privacy_satisfied", False)
    reidentification_risk = privacy_metrics.get("reidentification_risk", 1.0)
    sensitive_data = privacy_metrics.get("sensitive_data_found", True)
    
    # 모든 조건 만족 시 준수
    return (
        k_anonymity and
        differential_privacy and
        reidentification_risk < settings.MAX_REIDENTIFICATION_RISK and
        not sensitive_data
    )

def generate_recommendations(
    validation_results: Dict[str, Any],
    overall_quality: float,
    overall_bias: float
) -> List[str]:
    """개선 권장사항 생성"""
    
    recommendations = []
    
    # 품질 관련 권장사항
    if overall_quality < settings.MIN_QUALITY_SCORE:
        recommendations.append(
            f"품질 점수({overall_quality:.2f})가 최소 기준({settings.MIN_QUALITY_SCORE})보다 낮습니다. "
            "데이터 생성 파라미터를 조정하거나 더 많은 생성기를 사용해보세요."
        )
    
    quality_metrics = validation_results.get("quality_metrics", {})
    
    if quality_metrics.get("uniqueness", 1.0) < 0.7:
        recommendations.append(
            "데이터 다양성이 부족합니다. 생성 온도(temperature)를 높이거나 "
            "다양성 강화 생성기를 추가해보세요."
        )
    
    if quality_metrics.get("semantic_coherence", 1.0) < 0.5:
        recommendations.append(
            "의미적 일관성이 낮습니다. 프롬프트를 더 명확하게 작성하거나 "
            "도메인별 제약사항을 강화해보세요."
        )
    
    # 편향성 관련 권장사항
    if overall_bias > settings.MAX_BIAS_SCORE:
        recommendations.append(
            f"편향성 점수({overall_bias:.2f})가 최대 허용치({settings.MAX_BIAS_SCORE})를 초과합니다. "
            "편향성 완화 설정을 강화하세요."
        )
    
    bias_metrics = validation_results.get("bias_metrics", {})
    
    if "mitigation_suggestions" in bias_metrics:
        recommendations.extend(bias_metrics["mitigation_suggestions"])
    
    # 개인정보보호 관련 권장사항
    privacy_metrics = validation_results.get("privacy_metrics", {})
    
    if privacy_metrics.get("sensitive_data_found", False):
        recommendations.append(
            "민감한 개인정보가 발견되었습니다. 도메인 제약사항과 마스킹 규칙을 확인하세요."
        )
    
    if privacy_metrics.get("reidentification_risk", 0) > settings.MAX_REIDENTIFICATION_RISK:
        recommendations.append(
            "재식별 위험이 높습니다. k-익명성 파라미터를 높이거나 "
            "차등 개인정보보호 설정을 강화하세요."
        )
    
    if not recommendations:
        recommendations.append("전반적으로 양호한 품질을 유지하고 있습니다.")
    
    return recommendations

def generate_warnings(
    validation_results: Dict[str, Any],
    domain: str
) -> List[str]:
    """경고사항 생성"""
    
    warnings = []
    
    # 도메인별 경고
    domain_config = settings.DOMAINS.get(domain, {})
    
    quality_metrics = validation_results.get("quality_metrics", {})
    overall_quality = quality_metrics.get("overall_quality_score", 0)
    
    min_quality = domain_config.get("min_quality", 0.85)
    if overall_quality < min_quality:
        warnings.append(
            f"{domain} 도메인의 최소 품질 기준({min_quality})을 만족하지 못합니다."
        )
    
    # 준수사항 경고
    compliance = domain_config.get("compliance", [])
    privacy_metrics = validation_results.get("privacy_metrics", {})
    violations = privacy_metrics.get("privacy_violations", [])
    
    if "HIPAA" in compliance and domain == "medical":
        if violations:
            warnings.append(
                "HIPAA 준수 위반 가능성이 있습니다. PHI 마스킹을 확인하세요."
            )
    
    if "GDPR" in compliance:
        if privacy_metrics.get("sensitive_data_found", False):
            warnings.append(
                "GDPR 준수를 위해 개인정보를 추가로 익명화해야 합니다."
            )
    
    # 편향성 경고
    bias_metrics = validation_results.get("bias_metrics", {})
    
    if "category_biases" in bias_metrics:
        for category, biases in bias_metrics["category_biases"].items():
            if biases.get("underrepresentation", 0) > 0.3:
                warnings.append(
                    f"{category} 범주에서 특정 그룹이 과소표현되고 있습니다."
                )
    
    return warnings

def calculate_regex_metric(samples: List[Any], pattern: str) -> Dict[str, Any]:
    """정규식 기반 메트릭 계산"""
    
    import re
    
    matches = 0
    total = len(samples)
    
    for sample in samples:
        content = str(sample.content)
        if re.search(pattern, content):
            matches += 1
    
    return {
        "pattern": pattern,
        "matches": matches,
        "total": total,
        "match_rate": matches / total if total > 0 else 0
    }

def calculate_statistical_metric(samples: List[Any], method: str) -> Dict[str, Any]:
    """통계적 메트릭 계산"""
    
    values = []
    
    for sample in samples:
        content = sample.content
        if isinstance(content, str):
            values.append(len(content))
        elif isinstance(content, (int, float)):
            values.append(content)
    
    if not values:
        return {"error": "No numeric values found"}
    
    result = {
        "method": method,
        "count": len(values)
    }
    
    if method == "mean":
        result["value"] = np.mean(values)
    elif method == "median":
        result["value"] = np.median(values)
    elif method == "std":
        result["value"] = np.std(values)
    elif method == "range":
        result["value"] = np.max(values) - np.min(values)
    else:
        result["error"] = f"Unknown method: {method}"
    
    return result

def calculate_threshold_metric(
    samples: List[Any],
    field: str,
    threshold: float
) -> Dict[str, Any]:
    """임계값 기반 메트릭 계산"""
    
    above_threshold = 0
    below_threshold = 0
    
    for sample in samples:
        value = None
        
        if field == "quality_score":
            value = sample.quality_score
        elif field == "length":
            value = len(str(sample.content))
        
        if value is not None:
            if value >= threshold:
                above_threshold += 1
            else:
                below_threshold += 1
    
    total = above_threshold + below_threshold
    
    return {
        "field": field,
        "threshold": threshold,
        "above_threshold": above_threshold,
        "below_threshold": below_threshold,
        "pass_rate": above_threshold / total if total > 0 else 0
    }