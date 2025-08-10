from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from datetime import datetime
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from app.config import settings

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """기본 생성기 추상 클래스"""
    
    def __init__(
        self, 
        generator_id: str,
        model_name: str,
        generator_type: str = "text",
        config: Dict[str, Any] = None
    ):
        self.generator_id = generator_id
        self.model_name = model_name
        self.generator_type = generator_type
        self.config = config or {}
        self.is_initialized = False
        self.performance_metrics = {
            "total_generations": 0,
            "average_quality": 0.0,
            "average_bias": 0.0,
            "average_time": 0.0,
            "success_rate": 1.0
        }
        self.weight = 0.2  # 초기 가중치
        
    @abstractmethod
    async def initialize(self):
        """생성기 초기화"""
        pass
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        quantity: int,
        constraints: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """데이터 생성"""
        pass
    
    @abstractmethod
    async def validate_output(
        self, 
        generated_data: List[Dict[str, Any]]
    ) -> Tuple[bool, float]:
        """생성된 데이터 검증"""
        pass
    
    async def cleanup(self):
        """리소스 정리"""
        self.is_initialized = False
        logger.info(f"Generator {self.generator_id} cleaned up")
    
    def update_metrics(
        self, 
        quality_score: float, 
        bias_score: float, 
        generation_time: float
    ):
        """성능 메트릭 업데이트"""
        n = self.performance_metrics["total_generations"]
        
        # 이동 평균 계산
        self.performance_metrics["average_quality"] = (
            (self.performance_metrics["average_quality"] * n + quality_score) / (n + 1)
        )
        self.performance_metrics["average_bias"] = (
            (self.performance_metrics["average_bias"] * n + bias_score) / (n + 1)
        )
        self.performance_metrics["average_time"] = (
            (self.performance_metrics["average_time"] * n + generation_time) / (n + 1)
        )
        self.performance_metrics["total_generations"] = n + 1
        
        # 가중치 동적 조정
        self._adjust_weight()
    
    def _adjust_weight(self):
        """가중치 동적 조정"""
        # 품질과 편향성을 기반으로 가중치 조정
        quality_factor = self.performance_metrics["average_quality"]
        bias_factor = 1.0 - self.performance_metrics["average_bias"]
        
        # 가중치 계산 (품질 70%, 편향성 30%)
        self.weight = 0.7 * quality_factor + 0.3 * bias_factor
        self.weight = max(0.1, min(1.0, self.weight))  # 0.1 ~ 1.0 범위

class TransformerTextGenerator(BaseGenerator):
    """Transformer 기반 텍스트 생성기"""
    
    def __init__(self, generator_id: str, model_name: str, config: Dict[str, Any] = None):
        super().__init__(generator_id, model_name, "text", config)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """모델 초기화"""
        try:
            logger.info(f"Initializing {self.model_name} on {self.device}")
            
            # 모델과 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # GPU로 이동
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 파이프라인 생성
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            self.is_initialized = True
            logger.info(f"Successfully initialized {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name}: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        quantity: int,
        constraints: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """텍스트 데이터 생성"""
        if not self.is_initialized:
            await self.initialize()
        
        generated_data = []
        constraints = constraints or {}
        
        # 생성 파라미터 설정
        generation_params = {
            "max_length": constraints.get("max_length", 200),
            "min_length": constraints.get("min_length", 10),
            "temperature": constraints.get("temperature", 0.8),
            "top_p": constraints.get("top_p", 0.9),
            "do_sample": True,
            "num_return_sequences": min(quantity, 10),  # 배치 처리
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # 도메인별 프롬프트 강화
        domain = constraints.get("domain", "general")
        enhanced_prompt = self._enhance_prompt_for_domain(prompt, domain)
        
        # 편향성 제어 토큰 추가
        if constraints.get("bias_mitigation", {}).get("demographic_balance"):
            enhanced_prompt = self._add_bias_control_tokens(enhanced_prompt)
        
        try:
            start_time = datetime.now()
            
            # 배치 단위로 생성
            num_batches = (quantity + 9) // 10  # 10개씩 배치 처리
            
            for batch_idx in range(num_batches):
                current_batch_size = min(10, quantity - len(generated_data))
                generation_params["num_return_sequences"] = current_batch_size
                
                # 생성 실행
                outputs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.pipeline(
                        enhanced_prompt,
                        **generation_params
                    )
                )
                
                # 결과 처리
                for output in outputs:
                    generated_text = output["generated_text"]
                    
                    # 프롬프트 제거
                    if generated_text.startswith(enhanced_prompt):
                        generated_text = generated_text[len(enhanced_prompt):].strip()
                    
                    # 후처리
                    processed_text = self._post_process_text(generated_text, constraints)
                    
                    generated_data.append({
                        "content": processed_text,
                        "metadata": {
                            "generator_id": self.generator_id,
                            "model": self.model_name,
                            "domain": domain,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                
                # 진행 상황 로그
                logger.info(f"Generated batch {batch_idx + 1}/{num_batches}")
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # 메트릭 업데이트
            quality_score = await self._calculate_quality_score(generated_data)
            bias_score = await self._calculate_bias_score(generated_data)
            self.update_metrics(quality_score, bias_score, generation_time)
            
            logger.info(f"Generated {len(generated_data)} samples in {generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        
        return generated_data
    
    async def validate_output(
        self,
        generated_data: List[Dict[str, Any]]
    ) -> Tuple[bool, float]:
        """생성된 데이터 검증"""
        if not generated_data:
            return False, 0.0
        
        valid_samples = 0
        quality_scores = []
        
        for sample in generated_data:
            content = sample.get("content", "")
            
            # 기본 검증
            if len(content) < 10:  # 최소 길이 체크
                continue
            
            # 중복 체크
            if self._is_repetitive(content):
                continue
            
            # 품질 점수 계산
            quality = self._calculate_text_quality(content)
            quality_scores.append(quality)
            
            if quality >= settings.MIN_QUALITY_SCORE:
                valid_samples += 1
        
        validity_ratio = valid_samples / len(generated_data)
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return validity_ratio >= 0.8, avg_quality
    
    def _enhance_prompt_for_domain(self, prompt: str, domain: str) -> str:
        """도메인별 프롬프트 강화"""
        domain_prefixes = {
            "medical": "In a medical context, following HIPAA guidelines, ",
            "financial": "In a financial context, following regulatory compliance, ",
            "legal": "In a legal context, maintaining professional standards, ",
            "general": ""
        }
        
        prefix = domain_prefixes.get(domain, "")
        return f"{prefix}{prompt}"
    
    def _add_bias_control_tokens(self, prompt: str) -> str:
        """편향성 제어 토큰 추가"""
        bias_control = "[BALANCED] [DIVERSE] [INCLUSIVE] "
        return f"{bias_control}{prompt}"
    
    def _post_process_text(
        self, 
        text: str, 
        constraints: Dict[str, Any]
    ) -> str:
        """텍스트 후처리"""
        # 공백 정리
        text = " ".join(text.split())
        
        # 길이 제한
        max_length = constraints.get("max_length", 500)
        if len(text) > max_length:
            text = text[:max_length].rsplit(" ", 1)[0] + "."
        
        # 도메인별 민감 정보 마스킹
        domain = constraints.get("domain", "general")
        if domain == "medical":
            text = self._mask_medical_info(text)
        elif domain == "financial":
            text = self._mask_financial_info(text)
        
        return text
    
    def _mask_medical_info(self, text: str) -> str:
        """의료 정보 마스킹"""
        import re
        # SSN 패턴 마스킹
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        # 날짜 일반화
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]', text)
        return text
    
    def _mask_financial_info(self, text: str) -> str:
        """금융 정보 마스킹"""
        import re
        # 신용카드 번호 패턴 마스킹
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)
        # 계좌번호 패턴 마스킹
        text = re.sub(r'\b\d{10,}\b', '[ACCOUNT]', text)
        return text
    
    def _is_repetitive(self, text: str) -> bool:
        """반복성 체크"""
        words = text.split()
        if len(words) < 10:
            return False
        
        # 단어 반복 비율 계산
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        
        return repetition_ratio > 0.7  # 70% 이상 반복
    
    def _calculate_text_quality(self, text: str) -> float:
        """텍스트 품질 점수 계산"""
        score = 1.0
        
        # 길이 점수
        length = len(text.split())
        if length < 5:
            score *= 0.5
        elif length > 200:
            score *= 0.9
        
        # 다양성 점수
        words = text.split()
        if words:
            diversity = len(set(words)) / len(words)
            score *= diversity
        
        # 구두점 존재 여부
        if not any(p in text for p in ['.', '!', '?']):
            score *= 0.8
        
        return min(1.0, max(0.0, score))
    
    async def _calculate_quality_score(
        self, 
        generated_data: List[Dict[str, Any]]
    ) -> float:
        """전체 품질 점수 계산"""
        if not generated_data:
            return 0.0
        
        scores = [
            self._calculate_text_quality(item["content"]) 
            for item in generated_data
        ]
        
        return np.mean(scores)
    
    async def _calculate_bias_score(
        self, 
        generated_data: List[Dict[str, Any]]
    ) -> float:
        """편향성 점수 계산 (0: 편향 없음, 1: 높은 편향)"""
        # 간단한 편향성 체크 (실제로는 더 정교한 메트릭 필요)
        bias_keywords = {
            "gender": ["he", "she", "man", "woman", "boy", "girl"],
            "age": ["young", "old", "elderly", "youth"],
            "race": ["white", "black", "asian", "hispanic"]
        }
        
        total_bias = 0.0
        
        for item in generated_data:
            text = item["content"].lower()
            bias_count = 0
            
            for category, keywords in bias_keywords.items():
                category_count = sum(text.count(kw) for kw in keywords)
                if category_count > 0:
                    # 균형 체크
                    distribution = [text.count(kw) for kw in keywords]
                    if max(distribution) > sum(distribution) * 0.7:
                        bias_count += 1
            
            item_bias = bias_count / len(bias_keywords)
            total_bias += item_bias
        
        return total_bias / len(generated_data) if generated_data else 0.0