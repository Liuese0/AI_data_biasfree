import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import random
import string
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_id(prefix: str = "") -> str:
    """고유 ID 생성"""
    import uuid
    unique_id = str(uuid.uuid4())
    
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id

def hash_content(content: str) -> str:
    """콘텐츠 해시 생성"""
    return hashlib.sha256(content.encode()).hexdigest()

def sanitize_input(text: str) -> str:
    """입력 텍스트 정제"""
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 특수 문자 이스케이프
    text = re.sub(r'[<>&"\'`]', '', text)
    
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "..."
) -> str:
    """텍스트 자르기"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_datetime(dt: datetime, format: str = "iso") -> str:
    """날짜시간 포맷팅"""
    if format == "iso":
        return dt.isoformat()
    elif format == "human":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif format == "date":
        return dt.strftime("%Y-%m-%d")
    else:
        return str(dt)

def parse_duration(duration_str: str) -> timedelta:
    """기간 문자열 파싱"""
    patterns = {
        r'(\d+)s': lambda x: timedelta(seconds=int(x)),
        r'(\d+)m': lambda x: timedelta(minutes=int(x)),
        r'(\d+)h': lambda x: timedelta(hours=int(x)),
        r'(\d+)d': lambda x: timedelta(days=int(x)),
    }
    
    for pattern, func in patterns.items():
        match = re.match(pattern, duration_str)
        if match:
            return func(match.group(1))
    
    raise ValueError(f"Invalid duration format: {duration_str}")

def calculate_statistics(values: List[Union[int, float]]) -> Dict[str, float]:
    """통계 계산"""
    if not values:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0
        }
    
    return {
        "count": len(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "percentiles": {
            "25": np.percentile(values, 25),
            "50": np.percentile(values, 50),
            "75": np.percentile(values, 75),
            "90": np.percentile(values, 90),
            "95": np.percentile(values, 95)
        }
    }

def generate_random_string(
    length: int = 10,
    chars: str = string.ascii_letters + string.digits
) -> str:
    """랜덤 문자열 생성"""
    return ''.join(random.choice(chars) for _ in range(length))

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """리스트 청크 분할"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '.'
) -> Dict[str, Any]:
    """중첩 딕셔너리 평탄화"""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """딕셔너리 병합"""
    result = {}
    
    for d in dicts:
        result.update(d)
    
    return result

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """안전한 나눗셈"""
    if denominator == 0:
        return default
    return numerator / denominator

def normalize_values(
    values: List[float],
    min_val: float = 0,
    max_val: float = 1
) -> List[float]:
    """값 정규화"""
    if not values:
        return []
    
    current_min = min(values)
    current_max = max(values)
    
    if current_max == current_min:
        return [min_val] * len(values)
    
    scale = (max_val - min_val) / (current_max - current_min)
    
    return [
        min_val + (v - current_min) * scale
        for v in values
    ]

def extract_keywords(
    text: str,
    max_keywords: int = 10
) -> List[str]:
    """키워드 추출 (간단한 버전)"""
    # 불용어
    stopwords = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as',
        'are', 'was', 'were', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'could', 'to', 'of',
        'in', 'for', 'with', 'by', 'from', 'about', 'into',
        'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'along', 'following',
        'behind', 'beyond', 'within', 'without', 'around'
    }
    
    # 단어 추출
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    # 불용어 제거
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # 빈도 계산
    from collections import Counter
    word_freq = Counter(words)
    
    # 상위 키워드 반환
    return [word for word, _ in word_freq.most_common(max_keywords)]

def validate_email(email: str) -> bool:
    """이메일 검증"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """URL 검증"""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))

def format_file_size(size_bytes: int) -> str:
    """파일 크기 포맷팅"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"

def calculate_similarity(text1: str, text2: str) -> float:
    """텍스트 유사도 계산 (Jaccard)"""
    if not text1 or not text2:
        return 0.0
    
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    if not set1 and not set2:
        return 1.0
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union)

def retry_with_backoff(
    func,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0
):
    """지수 백오프로 재시도"""
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            
            import time
            time.sleep(delay)
            delay *= backoff_factor

def clean_json_string(json_str: str) -> str:
    """JSON 문자열 정리"""
    # 주석 제거
    json_str = re.sub(r'//.*?\n', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # 후행 쉼표 제거
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    return json_str

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """안전한 JSON 파싱"""
    try:
        cleaned = clean_json_string(json_str)
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {json_str[:100]}...")
        return default

class Timer:
    """실행 시간 측정"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"{self.name} took {self.elapsed:.2f} seconds")