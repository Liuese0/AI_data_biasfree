import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import re
import logging
from datetime import datetime
from sklearn.metrics import confusion_matrix
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

class BiasMetrics:
    """편향성 메트릭 계산"""
    
    def __init__(self):
        self.bias_categories = {
            "gender": {
                "keywords": {
                    "male": ["he", "him", "his", "man", "men", "boy", "male", "gentleman", "father", "son"],
                    "female": ["she", "her", "hers", "woman", "women", "girl", "female", "lady", "mother", "daughter"],
                    "neutral": ["they", "them", "their", "person", "people", "individual", "human"]
                },
                "weight": 0.25
            },
            "age": {
                "keywords": {
                    "young": ["young", "youth", "child", "kid", "teen", "teenager", "adolescent"],
                    "middle": ["adult", "middle-aged", "mature"],
                    "old": ["old", "elderly", "senior", "aged", "elder"]
                },
                "weight": 0.20
            },
            "ethnicity": {
                "keywords": {
                    "general": ["white", "black", "asian", "hispanic", "latino", "african", "european", "american"]
                },
                "weight": 0.25
            },
            "socioeconomic": {
                "keywords": {
                    "high": ["wealthy", "rich", "affluent", "privileged", "elite"],
                    "middle": ["middle-class", "average", "moderate"],
                    "low": ["poor", "underprivileged", "disadvantaged", "low-income"]
                },
                "weight": 0.15
            },
            "occupation": {
                "keywords": {
                    "professional": ["doctor", "lawyer", "engineer", "scientist", "professor", "ceo", "manager"],
                    "service": ["nurse", "teacher", "clerk", "assistant", "secretary"],
                    "labor": ["worker", "laborer", "mechanic", "driver", "farmer"]
                },
                "weight": 0.15
            }
        }
        
        self.metrics_history = []
        
    async def calculate_bias_metrics(
        self,
        generated_data: List[Dict[str, Any]],
        reference_data: Optional[List[Dict[str, Any]]] = None,
        protected_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """종합 편향성 메트릭 계산"""
        
        logger.info(f"Calculating bias metrics for {len(generated_data)} samples")
        
        metrics = {}
        
        # 기본 편향성 메트릭
        metrics["demographic_parity"] = await self._calculate_demographic_parity(
            generated_data, protected_attributes
        )
        
        metrics["representation_bias"] = self._calculate_representation_bias(generated_data)
        
        metrics["stereotype_bias"] = self._calculate_stereotype_bias(generated_data)
        
        metrics["linguistic_bias"] = self._calculate_linguistic_bias(generated_data)
        
        # 공정성 메트릭
        if reference_data:
            metrics["equalized_odds"] = await self._calculate_equalized_odds(
                generated_data, reference_data
            )
            
            metrics["disparate_impact"] = self._calculate_disparate_impact(
                generated_data, reference_data
            )
        
        # 범주별 편향성
        metrics["category_biases"] = self._calculate_category_biases(generated_data)
        
        # 종합 편향성 점수 (0: 편향 없음, 1: 높은 편향)
        metrics["overall_bias_score"] = self._calculate_overall_bias_score(metrics)
        
        # 편향성 완화 제안
        metrics["mitigation_suggestions"] = self._generate_mitigation_suggestions(metrics)
        
        # 히스토리 저장
        self._save_metrics_history(metrics)
        
        logger.info(f"Bias metrics calculated: overall score = {metrics['overall_bias_score']:.3f}")
        
        return metrics
    
    async def _calculate_demographic_parity(
        self,
        data: List[Dict[str, Any]],
        protected_attributes: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """인구통계학적 패리티 계산"""
        
        if not protected_attributes:
            protected_attributes = ["gender", "age", "ethnicity"]
        
        parity_scores = {}
        
        for attribute in protected_attributes:
            if attribute in self.bias_categories:
                distribution = self._get_attribute_distribution(data, attribute)
                parity_score = self._calculate_parity_score(distribution)
                parity_scores[attribute] = parity_score
        
        # 전체 패리티 점수
        if parity_scores:
            parity_scores["overall"] = np.mean(list(parity_scores.values()))
        else:
            parity_scores["overall"] = 1.0  # 완벽한 패리티
        
        return parity_scores
    
    def _calculate_representation_bias(self, data: List[Dict[str, Any]]) -> float:
        """표현 편향성 계산"""
        
        representations = defaultdict(int)
        total_mentions = 0
        
        for item in data:
            content = str(item.get("content", "")).lower()
            
            for category, info in self.bias_categories.items():
                for group, keywords in info["keywords"].items():
                    if isinstance(keywords, list):
                        for keyword in keywords:
                            count = content.count(keyword)
                            if count > 0:
                                representations[f"{category}_{group}"] += count
                                total_mentions += count
        
        if total_mentions == 0:
            return 0.0  # 편향 없음
        
        # 엔트로피 계산 (다양성 측정)
        probabilities = [count / total_mentions for count in representations.values()]
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)
        
        # 최대 엔트로피
        max_entropy = np.log(len(representations)) if representations else 1
        
        # 편향성 점수 (낮은 엔트로피 = 높은 편향)
        if max_entropy > 0:
            diversity = entropy / max_entropy
            bias_score = 1.0 - diversity
        else:
            bias_score = 0.0
        
        return bias_score
    
    def _calculate_stereotype_bias(self, data: List[Dict[str, Any]]) -> float:
        """고정관념 편향성 계산"""
        
        stereotype_patterns = [
            # 성별 고정관념
            (r"women?.{0,20}(emotional|sensitive|caring)", 0.3),
            (r"men?.{0,20}(strong|leader|aggressive)", 0.3),
            (r"nurse.{0,20}(she|her|woman)", 0.2),
            (r"engineer.{0,20}(he|him|man)", 0.2),
            
            # 연령 고정관념
            (r"elderly.{0,20}(slow|frail|confused)", 0.3),
            (r"young.{0,20}(inexperienced|naive|reckless)", 0.3),
            
            # 직업 고정관념
            (r"secretary.{0,20}(she|her|woman)", 0.2),
            (r"CEO.{0,20}(he|him|man)", 0.2),
        ]
        
        total_bias = 0.0
        pattern_count = 0
        
        for item in data:
            content = str(item.get("content", "")).lower()
            
            for pattern, weight in stereotype_patterns:
                if re.search(pattern, content):
                    total_bias += weight
                    pattern_count += 1
        
        if len(data) > 0:
            # 정규화
            bias_score = total_bias / len(data)
            return min(1.0, bias_score)
        
        return 0.0
    
    def _calculate_linguistic_bias(self, data: List[Dict[str, Any]]) -> float:
        """언어적 편향성 계산"""
        
        biased_language_patterns = [
            # 배타적 언어
            (r"\b(mankind|manpower|chairman|policeman)\b", "gendered_terms"),
            (r"\b(crazy|insane|nuts|psycho)\b", "mental_health"),
            (r"\b(lame|dumb|blind to)\b", "ability"),
            
            # 주관적 가정
            (r"\b(obviously|clearly|everyone knows|normal people)\b", "assumptions"),
            
            # 과도한 일반화
            (r"\b(all|every|never|always|none)\b.{0,20}(women|men|asian|black|white)", "overgeneralization")
        ]
        
        bias_counts = defaultdict(int)
        total_texts = len(data)
        
        for item in data:
            content = str(item.get("content", "")).lower()
            
            for pattern, bias_type in biased_language_patterns:
                if re.search(pattern, content):
                    bias_counts[bias_type] += 1
        
        if total_texts > 0:
            # 각 유형별 편향 비율
            bias_ratios = [count / total_texts for count in bias_counts.values()]
            
            # 전체 편향성 점수
            if bias_ratios:
                bias_score = np.mean(bias_ratios)
            else:
                bias_score = 0.0
            
            return min(1.0, bias_score)
        
        return 0.0
    
    async def _calculate_equalized_odds(
        self,
        generated_data: List[Dict[str, Any]],
        reference_data: List[Dict[str, Any]]
    ) -> float:
        """균등 기회 계산"""
        
        # 간단한 구현: 속성 분포 비교
        gen_distributions = {}
        ref_distributions = {}
        
        for category in ["gender", "age", "ethnicity"]:
            gen_distributions[category] = self._get_attribute_distribution(
                generated_data, category
            )
            ref_distributions[category] = self._get_attribute_distribution(
                reference_data, category
            )
        
        # 분포 차이 계산
        differences = []
        
        for category in gen_distributions:
            gen_dist = gen_distributions[category]
            ref_dist = ref_distributions[category]
            
            # KL divergence
            for key in set(gen_dist.keys()).union(set(ref_dist.keys())):
                gen_prob = gen_dist.get(key, 0.001)
                ref_prob = ref_dist.get(key, 0.001)
                
                if gen_prob > 0 and ref_prob > 0:
                    diff = abs(gen_prob - ref_prob)
                    differences.append(diff)
        
        if differences:
            # 차이가 작을수록 좋음 (1 - 평균 차이)
            equalized_odds = 1.0 - min(1.0, np.mean(differences))
        else:
            equalized_odds = 1.0
        
        return equalized_odds
    
    def _calculate_disparate_impact(
        self,
        generated_data: List[Dict[str, Any]],
        reference_data: List[Dict[str, Any]]
    ) -> float:
        """차별적 영향 비율 계산"""
        
        # 80% 규칙 적용
        gen_positive_rates = self._calculate_positive_rates(generated_data)
        ref_positive_rates = self._calculate_positive_rates(reference_data)
        
        if not gen_positive_rates or not ref_positive_rates:
            return 1.0  # 영향 없음
        
        disparate_impacts = []
        
        for attribute in gen_positive_rates:
            if attribute in ref_positive_rates:
                gen_rate = gen_positive_rates[attribute]
                ref_rate = ref_positive_rates[attribute]
                
                if ref_rate > 0:
                    impact_ratio = gen_rate / ref_rate
                    # 80% 규칙: 0.8 <= ratio <= 1.25
                    if 0.8 <= impact_ratio <= 1.25:
                        disparate_impacts.append(1.0)  # 공정
                    else:
                        # 벗어난 정도에 따라 점수 감소
                        if impact_ratio < 0.8:
                            disparate_impacts.append(impact_ratio / 0.8)
                        else:
                            disparate_impacts.append(1.25 / impact_ratio)
        
        if disparate_impacts:
            return np.mean(disparate_impacts)
        
        return 1.0
    
    def _calculate_category_biases(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """범주별 편향성 계산"""
        
        category_biases = {}
        
        for category, info in self.bias_categories.items():
            distribution = self._get_attribute_distribution(data, category)
            
            # 균형성 점수
            balance_score = self._calculate_balance_score(distribution)
            
            # 다양성 점수
            diversity_score = self._calculate_diversity_score(distribution)
            
            # 과소표현 체크
            underrepresentation = self._check_underrepresentation(distribution)
            
            category_biases[category] = {
                "balance": balance_score,
                "diversity": diversity_score,
                "underrepresentation": underrepresentation,
                "overall": np.mean([balance_score, diversity_score, 1.0 - underrepresentation])
            }
        
        return category_biases
    
    def _get_attribute_distribution(
        self,
        data: List[Dict[str, Any]],
        attribute: str
    ) -> Dict[str, float]:
        """속성 분포 계산"""
        
        if attribute not in self.bias_categories:
            return {}
        
        counts = defaultdict(int)
        total = 0
        
        for item in data:
            content = str(item.get("content", "")).lower()
            
            for group, keywords in self.bias_categories[attribute]["keywords"].items():
                if isinstance(keywords, list):
                    for keyword in keywords:
                        if keyword in content:
                            counts[group] += 1
                            total += 1
                            break  # 그룹당 한 번만 카운트
        
        # 정규화
        if total > 0:
            distribution = {group: count / total for group, count in counts.items()}
        else:
            distribution = {}
        
        return distribution
    
    def _calculate_parity_score(self, distribution: Dict[str, float]) -> float:
        """패리티 점수 계산"""
        
        if not distribution:
            return 1.0  # 완벽한 패리티 (데이터 없음)
        
        values = list(distribution.values())
        
        if len(values) < 2:
            return 1.0
        
        # 표준편차가 낮을수록 패리티가 좋음
        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        if mean_val > 0:
            cv = std_dev / mean_val  # 변동계수
            parity_score = 1.0 - min(1.0, cv)
        else:
            parity_score = 1.0
        
        return parity_score
    
    def _calculate_balance_score(self, distribution: Dict[str, float]) -> float:
        """균형성 점수 계산"""
        
        if not distribution:
            return 1.0
        
        # 이상적인 균등 분포와의 차이
        ideal_prob = 1.0 / len(distribution)
        differences = [abs(prob - ideal_prob) for prob in distribution.values()]
        
        # 차이가 작을수록 균형적
        avg_difference = np.mean(differences)
        balance_score = 1.0 - min(1.0, avg_difference * len(distribution))
        
        return max(0.0, balance_score)
    
    def _calculate_diversity_score(self, distribution: Dict[str, float]) -> float:
        """다양성 점수 계산 (Shannon 엔트로피)"""
        
        if not distribution:
            return 0.0
        
        # Shannon 엔트로피
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in distribution.values())
        
        # 최대 엔트로피 (균등 분포)
        max_entropy = np.log(len(distribution))
        
        if max_entropy > 0:
            diversity_score = entropy / max_entropy
        else:
            diversity_score = 0.0
        
        return diversity_score
    
    def _check_underrepresentation(self, distribution: Dict[str, float]) -> float:
        """과소표현 체크"""
        
        if not distribution:
            return 0.0
        
        # 임계값 이하 그룹 찾기
        threshold = 0.1  # 10% 미만을 과소표현으로 간주
        underrepresented = [prob for prob in distribution.values() if prob < threshold]
        
        if len(distribution) > 0:
            underrep_ratio = len(underrepresented) / len(distribution)
        else:
            underrep_ratio = 0.0
        
        return underrep_ratio
    
    def _calculate_positive_rates(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """긍정적 결과 비율 계산"""
        
        positive_keywords = ["success", "good", "excellent", "positive", "achieve", "win"]
        rates = {}
        
        for category in ["gender", "age", "ethnicity"]:
            distribution = self._get_attribute_distribution(data, category)
            
            for group in distribution:
                positive_count = 0
                total_count = 0
                
                for item in data:
                    content = str(item.get("content", "")).lower()
                    
                    # 그룹 키워드 체크
                    group_keywords = self.bias_categories[category]["keywords"].get(group, [])
                    if any(kw in content for kw in group_keywords):
                        total_count += 1
                        
                        # 긍정적 키워드 체크
                        if any(pk in content for pk in positive_keywords):
                            positive_count += 1
                
                if total_count > 0:
                    rates[f"{category}_{group}"] = positive_count / total_count
        
        return rates
    
    def _calculate_overall_bias_score(self, metrics: Dict[str, Any]) -> float:
        """종합 편향성 점수 계산"""
        
        scores = []
        weights = {
            "demographic_parity": 0.25,
            "representation_bias": 0.20,
            "stereotype_bias": 0.20,
            "linguistic_bias": 0.15,
            "equalized_odds": 0.10,
            "disparate_impact": 0.10
        }
        
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                
                if isinstance(value, dict) and "overall" in value:
                    # 패리티 점수는 높을수록 좋음 (편향 낮음)
                    score = 1.0 - value["overall"]
                elif metric in ["equalized_odds", "disparate_impact"]:
                    # 이 메트릭들은 높을수록 좋음
                    score = 1.0 - value
                else:
                    # 나머지는 낮을수록 좋음
                    score = value
                
                scores.append(score * weight)
        
        if scores:
            overall_bias = sum(scores) / sum(weights.values())
        else:
            overall_bias = 0.5  # 기본값
        
        return min(1.0, max(0.0, overall_bias))
    
    def _generate_mitigation_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """편향성 완화 제안 생성"""
        
        suggestions = []
        
        # 인구통계학적 패리티 체크
        if "demographic_parity" in metrics:
            parity = metrics["demographic_parity"]
            for attr, score in parity.items():
                if attr != "overall" and score < 0.7:
                    suggestions.append(
                        f"{attr} 속성의 균형을 개선하세요 (현재 패리티: {score:.2f})"
                    )
        
        # 표현 편향성 체크
        if metrics.get("representation_bias", 0) > 0.3:
            suggestions.append(
                "다양한 그룹의 표현을 균형있게 포함시키세요"
            )
        
        # 고정관념 편향성 체크
        if metrics.get("stereotype_bias", 0) > 0.3:
            suggestions.append(
                "고정관념을 강화하는 표현을 피하고 다양한 관점을 포함하세요"
            )
        
        # 언어적 편향성 체크
        if metrics.get("linguistic_bias", 0) > 0.3:
            suggestions.append(
                "포용적이고 중립적인 언어를 사용하세요"
            )
        
        # 범주별 제안
        if "category_biases" in metrics:
            for category, biases in metrics["category_biases"].items():
                if biases.get("underrepresentation", 0) > 0.3:
                    suggestions.append(
                        f"{category} 범주에서 과소표현된 그룹을 더 포함시키세요"
                    )
        
        if not suggestions:
            suggestions.append("전반적으로 양호한 편향성 수준을 유지하고 있습니다")
        
        return suggestions
    
    def _save_metrics_history(self, metrics: Dict[str, Any]):
        """메트릭 히스토리 저장"""
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy()
        })
        
        # 히스토리 크기 제한
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_bias_trends(self) -> Dict[str, Any]:
        """편향성 트렌드 분석"""
        
        if len(self.metrics_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_scores = [
            h["metrics"].get("overall_bias_score", 0.5)
            for h in self.metrics_history[-10:]
        ]
        
        older_scores = [
            h["metrics"].get("overall_bias_score", 0.5)
            for h in self.metrics_history[-20:-10]
        ] if len(self.metrics_history) >= 20 else recent_scores
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        improvement = older_avg - recent_avg  # 편향이 감소하면 양수
        
        return {
            "current_bias": recent_avg,
            "previous_bias": older_avg,
            "improvement": improvement,
            "trend": "improving" if improvement > 0.05 else "declining" if improvement < -0.05 else "stable"
        }