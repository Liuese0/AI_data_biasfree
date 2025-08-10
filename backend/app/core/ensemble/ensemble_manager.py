import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import logging
import random
from collections import defaultdict

from app.core.ensemble.base_generator import BaseGenerator
from app.core.ensemble.text_generators import (
    GPT2Generator,
    T5Generator,
    BERTGenerator,
    DiversityGenerator
)
from app.config import settings

logger = logging.getLogger(__name__)

class EnsembleManager:
    """앙상블 생성기 관리자"""
    
    def __init__(self):
        self.generators: Dict[str, BaseGenerator] = {}
        self.active_generators: List[str] = []
        self.is_ready = False
        self.total_generations = 0
        self.average_quality_score = 0.0
        self.average_bias_score = 0.0
        self.generation_history = []
        
    async def initialize(self):
        """앙상블 매니저 초기화"""
        logger.info("Initializing ensemble manager...")
        
        try:
            # 생성기 초기화
            await self._initialize_generators()
            
            # 생성기 검증
            await self._validate_generators()
            
            # 가중치 초기화
            self._initialize_weights()
            
            self.is_ready = True
            logger.info(f"Ensemble manager ready with {len(self.active_generators)} generators")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble manager: {e}")
            raise
    
    async def _initialize_generators(self):
        """생성기 초기화"""
        generator_configs = [
            ("gpt2_small", GPT2Generator, "gpt2"),
            ("gpt2_medium", GPT2Generator, "gpt2-medium"),
            ("t5_small", T5Generator, "t5-small"),
            ("bert_base", BERTGenerator, "bert-base-uncased"),
            ("diversity", DiversityGenerator, None)
        ]
        
        initialization_tasks = []
        
        for gen_id, gen_class, model_name in generator_configs:
            try:
                if gen_class == DiversityGenerator:
                    generator = gen_class(gen_id)
                else:
                    generator = gen_class(gen_id, model_name)
                
                self.generators[gen_id] = generator
                initialization_tasks.append(generator.initialize())
                
            except Exception as e:
                logger.warning(f"Failed to create generator {gen_id}: {e}")
        
        # 병렬 초기화
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # 성공적으로 초기화된 생성기만 활성화
        for i, (gen_id, _, _) in enumerate(generator_configs):
            if gen_id in self.generators:
                if isinstance(results[i], Exception):
                    logger.warning(f"Generator {gen_id} initialization failed: {results[i]}")
                    del self.generators[gen_id]
                else:
                    self.active_generators.append(gen_id)
                    logger.info(f"Generator {gen_id} activated")
    
    async def _validate_generators(self):
        """생성기 검증"""
        if len(self.active_generators) < settings.MIN_ENSEMBLE_GENERATORS:
            raise ValueError(
                f"Insufficient generators: {len(self.active_generators)} < {settings.MIN_ENSEMBLE_GENERATORS}"
            )
        
        # 각 생성기 테스트
        test_prompt = "Test generation"
        validation_tasks = []
        
        for gen_id in self.active_generators[:]:
            generator = self.generators[gen_id]
            validation_tasks.append(self._validate_single_generator(generator, test_prompt))
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # 실패한 생성기 제거
        for i, gen_id in enumerate(list(self.active_generators)):
            if isinstance(results[i], Exception) or not results[i]:
                logger.warning(f"Removing invalid generator {gen_id}")
                self.active_generators.remove(gen_id)
    
    async def _validate_single_generator(
        self, 
        generator: BaseGenerator, 
        test_prompt: str
    ) -> bool:
        """단일 생성기 검증"""
        try:
            result = await generator.generate(test_prompt, 1)
            if result and len(result) > 0:
                is_valid, quality = await generator.validate_output(result)
                return is_valid
        except Exception as e:
            logger.error(f"Generator validation failed: {e}")
        return False
    
    def _initialize_weights(self):
        """생성기 가중치 초기화"""
        num_generators = len(self.active_generators)
        if num_generators == 0:
            return
        
        # 균등 가중치로 시작
        initial_weight = 1.0 / num_generators
        
        for gen_id in self.active_generators:
            self.generators[gen_id].weight = initial_weight
        
        logger.info(f"Initialized weights for {num_generators} generators")
    
    async def generate_ensemble(
        self,
        prompt: str,
        quantity: int,
        domain: str = "general",
        constraints: Dict[str, Any] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """앙상블 생성 실행"""
        if not self.is_ready:
            raise RuntimeError("Ensemble manager not initialized")
        
        logger.info(f"Starting ensemble generation: {quantity} samples for domain '{domain}'")
        
        constraints = constraints or {}
        constraints["domain"] = domain
        
        # 생성기 선택
        selected_generators = self._select_generators(domain, constraints)
        
        if len(selected_generators) < settings.MIN_ENSEMBLE_GENERATORS:
            raise ValueError(f"Insufficient generators for ensemble: {len(selected_generators)}")
        
        # 각 생성기별 할당량 계산
        allocations = self._calculate_allocations(selected_generators, quantity)
        
        # 병렬 생성
        generation_tasks = []
        for gen_id, alloc_quantity in allocations.items():
            if alloc_quantity > 0:
                generator = self.generators[gen_id]
                task = self._generate_with_generator(
                    generator, 
                    prompt, 
                    alloc_quantity, 
                    constraints
                )
                generation_tasks.append((gen_id, task))
        
        # 결과 수집
        all_results = []
        generator_metrics = {}
        
        results = await asyncio.gather(
            *[task for _, task in generation_tasks],
            return_exceptions=True
        )
        
        for i, (gen_id, _) in enumerate(generation_tasks):
            if isinstance(results[i], Exception):
                logger.error(f"Generator {gen_id} failed: {results[i]}")
                generator_metrics[gen_id] = {"status": "failed", "error": str(results[i])}
            else:
                generated_data, metrics = results[i]
                all_results.extend(generated_data)
                generator_metrics[gen_id] = metrics
        
        # 결과 통합 및 선택
        final_results = await self._aggregate_results(all_results, quantity, constraints)
        
        # 앙상블 메트릭 계산
        ensemble_metrics = self._calculate_ensemble_metrics(final_results, generator_metrics)
        
        # 가중치 업데이트
        await self._update_weights(generator_metrics)
        
        # 통계 업데이트
        self._update_statistics(final_results, ensemble_metrics)
        
        logger.info(f"Ensemble generation completed: {len(final_results)} samples")
        
        return final_results, ensemble_metrics
    
    def _select_generators(
        self, 
        domain: str, 
        constraints: Dict[str, Any]
    ) -> List[str]:
        """도메인과 제약사항에 따른 생성기 선택"""
        selected = []
        
        # 도메인별 선호 생성기
        domain_preferences = {
            "medical": ["t5_small", "bert_base"],
            "financial": ["gpt2_medium", "t5_small"],
            "legal": ["gpt2_medium", "bert_base"],
            "general": self.active_generators
        }
        
        preferred = domain_preferences.get(domain, self.active_generators)
        
        # 활성 생성기 중에서 선택
        for gen_id in preferred:
            if gen_id in self.active_generators:
                selected.append(gen_id)
        
        # 다양성 생성기 추가
        if "diversity" in self.active_generators and constraints.get("bias_mitigation", {}).get("cultural_diversity"):
            if "diversity" not in selected:
                selected.append("diversity")
        
        # 최소 개수 보장
        if len(selected) < settings.MIN_ENSEMBLE_GENERATORS:
            for gen_id in self.active_generators:
                if gen_id not in selected:
                    selected.append(gen_id)
                if len(selected) >= settings.MIN_ENSEMBLE_GENERATORS:
                    break
        
        # 최대 개수 제한
        if len(selected) > settings.MAX_ENSEMBLE_GENERATORS:
            # 가중치 기준으로 상위 N개 선택
            selected.sort(key=lambda x: self.generators[x].weight, reverse=True)
            selected = selected[:settings.MAX_ENSEMBLE_GENERATORS]
        
        return selected
    
    def _calculate_allocations(
        self, 
        selected_generators: List[str], 
        total_quantity: int
    ) -> Dict[str, int]:
        """생성기별 생성량 할당"""
        allocations = {}
        
        # 가중치 기반 할당
        total_weight = sum(self.generators[gen_id].weight for gen_id in selected_generators)
        
        remaining = total_quantity
        for gen_id in selected_generators:
            weight = self.generators[gen_id].weight
            allocation = int(total_quantity * (weight / total_weight))
            allocations[gen_id] = allocation
            remaining -= allocation
        
        # 남은 수량 분배
        if remaining > 0:
            # 가장 높은 가중치를 가진 생성기에 할당
            best_generator = max(selected_generators, key=lambda x: self.generators[x].weight)
            allocations[best_generator] += remaining
        
        # 최소 생성량 보장
        min_per_generator = max(1, total_quantity // (len(selected_generators) * 2))
        for gen_id in selected_generators:
            if allocations[gen_id] < min_per_generator:
                allocations[gen_id] = min_per_generator
        
        return allocations
    
    async def _generate_with_generator(
        self,
        generator: BaseGenerator,
        prompt: str,
        quantity: int,
        constraints: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """단일 생성기로 생성"""
        start_time = datetime.now()
        
        try:
            # 생성 실행
            generated_data = await generator.generate(prompt, quantity, constraints)
            
            # 검증
            is_valid, quality_score = await generator.validate_output(generated_data)
            
            # 메트릭 계산
            generation_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                "generator_id": generator.generator_id,
                "quantity_requested": quantity,
                "quantity_generated": len(generated_data),
                "quality_score": quality_score,
                "generation_time": generation_time,
                "is_valid": is_valid,
                "status": "success"
            }
            
            return generated_data, metrics
            
        except Exception as e:
            logger.error(f"Generation failed for {generator.generator_id}: {e}")
            raise
    
    async def _aggregate_results(
        self,
        all_results: List[Dict[str, Any]],
        target_quantity: int,
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """결과 통합 및 선택"""
        if not all_results:
            return []
        
        # 품질 점수 계산
        for item in all_results:
            if "quality_score" not in item:
                item["quality_score"] = await self._calculate_item_quality(item, constraints)
        
        # 편향성 점수 계산
        bias_scores = await self._calculate_bias_scores(all_results, constraints)
        for i, item in enumerate(all_results):
            item["bias_score"] = bias_scores[i] if i < len(bias_scores) else 0.5
        
        # 종합 점수 계산 (품질 60%, 편향성 40%)
        for item in all_results:
            item["ensemble_score"] = (
                0.6 * item.get("quality_score", 0.5) +
                0.4 * (1.0 - item.get("bias_score", 0.5))
            )
        
        # 점수 기준 정렬
        all_results.sort(key=lambda x: x.get("ensemble_score", 0), reverse=True)
        
        # 상위 N개 선택
        selected = all_results[:target_quantity]
        
        # 다양성 보장
        selected = self._ensure_diversity(selected, all_results, target_quantity)
        
        return selected
    
    async def _calculate_item_quality(
        self, 
        item: Dict[str, Any], 
        constraints: Dict[str, Any]
    ) -> float:
        """개별 항목 품질 점수 계산"""
        content = item.get("content", "")
        
        if not content:
            return 0.0
        
        score = 1.0
        
        # 길이 체크
        if isinstance(content, str):
            word_count = len(content.split())
            if word_count < 5:
                score *= 0.5
            elif word_count > 500:
                score *= 0.9
        
        # 도메인별 체크
        domain = constraints.get("domain", "general")
        if domain == "medical":
            # 의료 용어 포함 여부
            medical_terms = ["patient", "treatment", "diagnosis", "symptoms", "medical"]
            if any(term in content.lower() for term in medical_terms):
                score *= 1.1
        
        return min(1.0, score)
    
    async def _calculate_bias_scores(
        self,
        results: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> List[float]:
        """편향성 점수 계산"""
        scores = []
        
        # 간단한 편향성 체크
        bias_indicators = {
            "gender": ["he", "she", "him", "her", "his", "hers"],
            "age": ["young", "old", "elderly", "child", "adult"],
            "ethnicity": ["white", "black", "asian", "hispanic", "ethnic"]
        }
        
        for item in results:
            content = str(item.get("content", "")).lower()
            bias_count = 0
            
            for category, keywords in bias_indicators.items():
                keyword_counts = [content.count(kw) for kw in keywords]
                if sum(keyword_counts) > 0:
                    # 불균형 체크
                    max_count = max(keyword_counts)
                    total_count = sum(keyword_counts)
                    if max_count > total_count * 0.6:  # 60% 이상 편중
                        bias_count += 1
            
            bias_score = bias_count / len(bias_indicators)
            scores.append(bias_score)
        
        return scores
    
    def _ensure_diversity(
        self,
        selected: List[Dict[str, Any]],
        all_results: List[Dict[str, Any]],
        target_quantity: int
    ) -> List[Dict[str, Any]]:
        """다양성 보장"""
        if len(selected) >= target_quantity:
            return selected[:target_quantity]
        
        # 생성기별 분포 확인
        generator_counts = defaultdict(int)
        for item in selected:
            gen_id = item.get("metadata", {}).get("generator_id", "unknown")
            generator_counts[gen_id] += 1
        
        # 부족한 생성기의 결과 추가
        needed = target_quantity - len(selected)
        
        for item in all_results:
            if item not in selected:
                gen_id = item.get("metadata", {}).get("generator_id", "unknown")
                
                # 적게 포함된 생성기 우선
                if generator_counts[gen_id] < target_quantity // len(self.active_generators):
                    selected.append(item)
                    generator_counts[gen_id] += 1
                    needed -= 1
                    
                    if needed <= 0:
                        break
        
        return selected[:target_quantity]
    
    def _calculate_ensemble_metrics(
        self,
        final_results: List[Dict[str, Any]],
        generator_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """앙상블 메트릭 계산"""
        metrics = {
            "total_generated": len(final_results),
            "generators_used": len(generator_metrics),
            "average_quality": np.mean([
                item.get("quality_score", 0) for item in final_results
            ]) if final_results else 0.0,
            "average_bias": np.mean([
                item.get("bias_score", 0) for item in final_results
            ]) if final_results else 0.0,
            "diversity_score": self._calculate_diversity_score(final_results),
            "generator_contributions": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # 생성기별 기여도
        for gen_id, gen_metrics in generator_metrics.items():
            if gen_metrics.get("status") == "success":
                metrics["generator_contributions"][gen_id] = {
                    "quantity": gen_metrics.get("quantity_generated", 0),
                    "quality": gen_metrics.get("quality_score", 0),
                    "time": gen_metrics.get("generation_time", 0)
                }
        
        return metrics
    
    def _calculate_diversity_score(self, results: List[Dict[str, Any]]) -> float:
        """결과 집합의 다양성 점수"""
        if not results:
            return 0.0
        
        # 생성기 다양성
        generators = set()
        for item in results:
            gen_id = item.get("metadata", {}).get("generator_id")
            if gen_id:
                generators.add(gen_id)
        
        generator_diversity = len(generators) / max(1, len(self.active_generators))
        
        # 콘텐츠 다양성 (간단한 버전)
        contents = [str(item.get("content", ""))[:100] for item in results]
        unique_starts = len(set(contents))
        content_diversity = unique_starts / len(contents) if contents else 0
        
        # 종합 다양성
        diversity = (generator_diversity + content_diversity) / 2
        
        return min(1.0, diversity)
    
    async def _update_weights(self, generator_metrics: Dict[str, Dict[str, Any]]):
        """생성기 가중치 업데이트"""
        for gen_id, metrics in generator_metrics.items():
            if gen_id not in self.generators:
                continue
            
            generator = self.generators[gen_id]
            
            if metrics.get("status") == "success":
                quality = metrics.get("quality_score", 0.5)
                
                # 성공률 반영
                success_rate = metrics.get("quantity_generated", 0) / max(1, metrics.get("quantity_requested", 1))
                
                # 새 가중치 계산
                performance_score = 0.7 * quality + 0.3 * success_rate
                
                # 지수 이동 평균
                alpha = 0.3
                generator.weight = alpha * performance_score + (1 - alpha) * generator.weight
                
            else:
                # 실패 시 가중치 감소
                generator.weight *= 0.9
            
            # 가중치 범위 제한
            generator.weight = max(0.1, min(1.0, generator.weight))
        
        # 가중치 정규화
        total_weight = sum(self.generators[gen_id].weight for gen_id in self.active_generators)
        if total_weight > 0:
            for gen_id in self.active_generators:
                self.generators[gen_id].weight /= total_weight
    
    def _update_statistics(
        self,
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ):
        """통계 업데이트"""
        self.total_generations += len(results)
        
        # 이동 평균 업데이트
        alpha = 0.1
        self.average_quality_score = (
            alpha * metrics.get("average_quality", 0) +
            (1 - alpha) * self.average_quality_score
        )
        self.average_bias_score = (
            alpha * metrics.get("average_bias", 0) +
            (1 - alpha) * self.average_bias_score
        )
        
        # 히스토리 추가
        self.generation_history.append({
            "timestamp": datetime.now(),
            "count": len(results),
            "quality": metrics.get("average_quality", 0),
            "bias": metrics.get("average_bias", 0),
            "diversity": metrics.get("diversity_score", 0)
        })
        
        # 히스토리 크기 제한
        if len(self.generation_history) > 1000:
            self.generation_history = self.generation_history[-1000:]
    
    @property
    def active_generators_count(self) -> int:
        """활성 생성기 수"""
        return len(self.active_generators)
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("Cleaning up ensemble manager...")
        
        cleanup_tasks = []
        for generator in self.generators.values():
            cleanup_tasks.append(generator.cleanup())
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.generators.clear()
        self.active_generators.clear()
        self.is_ready = False
        
        logger.info("Ensemble manager cleanup completed")