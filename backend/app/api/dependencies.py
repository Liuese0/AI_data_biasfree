from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import jwt
from datetime import datetime, timedelta
import logging

from app.config import settings
from app.models.database import get_db

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

class RateLimiter:
    """요청 속도 제한"""
    
    def __init__(self):
        self.requests = {}
        self.window_size = 60  # 1분
        self.max_requests = 100
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """속도 제한 체크"""
        now = datetime.utcnow()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # 오래된 요청 제거
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if (now - req_time).total_seconds() < self.window_size
        ]
        
        # 요청 수 체크
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # 새 요청 추가
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Optional[Dict[str, Any]]:
    """현재 사용자 조회 (선택적 인증)"""
    
    if not credentials:
        return None
    
    try:
        # JWT 토큰 디코드
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # 토큰 만료 체크
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            return None
        
        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "roles": payload.get("roles", [])
        }
        
    except jwt.PyJWTError:
        return None

async def require_auth(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """인증 필수"""
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return current_user

async def require_admin(
    current_user: Dict[str, Any] = Depends(require_auth)
) -> Dict[str, Any]:
    """관리자 권한 필수"""
    
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    
    return current_user

async def check_rate_limit_dependency(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """속도 제한 의존성"""
    
    # 사용자 ID 또는 "anonymous" 사용
    client_id = current_user["user_id"] if current_user else "anonymous"
    
    if not await rate_limiter.check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """액세스 토큰 생성"""
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt

class PaginationParams:
    """페이지네이션 파라미터"""
    
    def __init__(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 100
    ):
        if page < 1:
            raise HTTPException(
                status_code=400,
                detail="Page must be greater than 0"
            )
        
        if page_size < 1:
            raise HTTPException(
                status_code=400,
                detail="Page size must be greater than 0"
            )
        
        if page_size > max_page_size:
            raise HTTPException(
                status_code=400,
                detail=f"Page size cannot exceed {max_page_size}"
            )
        
        self.page = page
        self.page_size = page_size
        self.offset = (page - 1) * page_size
        self.limit = page_size

async def get_pagination(
    page: int = 1,
    page_size: int = 50
) -> PaginationParams:
    """페이지네이션 파라미터 조회"""
    return PaginationParams(page, page_size)

class QueryParams:
    """쿼리 파라미터"""
    
    def __init__(
        self,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        order: str = "desc",
        filters: Optional[Dict[str, Any]] = None
    ):
        self.search = search
        self.sort_by = sort_by
        self.order = order.lower()
        self.filters = filters or {}
        
        if self.order not in ["asc", "desc"]:
            raise HTTPException(
                status_code=400,
                detail="Order must be 'asc' or 'desc'"
            )

async def validate_domain(domain: str) -> str:
    """도메인 검증"""
    
    valid_domains = list(settings.DOMAINS.keys())
    
    if domain not in valid_domains:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain. Must be one of: {valid_domains}"
        )
    
    return domain

async def validate_job_ownership(
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
) -> bool:
    """작업 소유권 검증"""
    
    from app.models.database import JobRepository
    
    job = await JobRepository.get_job(db, job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 관리자는 모든 작업 접근 가능
    if "admin" in current_user.get("roles", []):
        return True
    
    # 소유자 체크 (실제 구현 시 job에 user_id 필드 필요)
    # if job.user_id != current_user["user_id"]:
    #     raise HTTPException(status_code=403, detail="Access denied")
    
    return True

class FeatureFlags:
    """기능 플래그"""
    
    def __init__(self):
        self.flags = {
            "image_generation": False,  # Phase 2
            "advanced_constraints": True,
            "custom_models": False,
            "batch_processing": True,
            "real_time_validation": True
        }
    
    def is_enabled(self, feature: str) -> bool:
        """기능 활성화 여부"""
        return self.flags.get(feature, False)
    
    def require_feature(self, feature: str):
        """기능 필수 체크"""
        if not self.is_enabled(feature):
            raise HTTPException(
                status_code=501,
                detail=f"Feature '{feature}' is not yet implemented"
            )

feature_flags = FeatureFlags()

async def require_feature(feature: str):
    """기능 플래그 의존성"""
    feature_flags.require_feature(feature)