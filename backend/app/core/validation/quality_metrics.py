import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from datetime import datetime
import logging
import re
from collections import Counter

from app.config import settings

logger = logging.getLogger(__name__)

class QualityMetrics:
    """데이터 품질 메트릭 계산"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.reference_data = None
        self.metrics_history = []
        
    async def calculate_quality_score(
        self,
        generated_data: List[Dict[str, Any]],
        reference_data: Optional[List[Dict[str, Any]]] = None,
        domain: str = "general"
    ) -> Dict[str, float]:
        """종합 품질 점수 계산"""
        
        logger.info(f"Calculating quality metrics for {len(generated_data)} samples")
        
        metrics = {}
        
        # 기본 품질 메트릭
        metrics["completeness"] = self._calculate_completeness(generated_data)
        metrics["consistency"] = self._calculate_consistency(generated_data)
        metrics["uniqueness"] = self._calculate_uniqueness(generated_data)
        metrics["validity"] = self._calculate_validity(generated_data, domain)
        
        # 통계적 품질 메트릭
        if reference_data:
            metrics["distribution_similarity"] = await self._calculate_distribution_similarity(
                generated_data, reference_data
            )
            metrics["statistical_parity"] = self._calculate_statistical_parity(
                generated_data, reference_data
            )
        
        # 텍스트 품질 메트릭
        if self._is_text_data(generated_data):
            text_metrics = await self._calculate_text_quality_metrics(generated_data, reference_data)
            metrics.update(text_metrics)
        
        # 종합 점수 계산
        metrics["overall_quality_score"] = self._calculate_overall_score(metrics)
        
        # 메트릭 히스토리 저장
        self._save_metrics_history(metrics)
        
        logger.info(f"Quality metrics calculated: overall score = {metrics['overall_quality_score']:.3f}")
        
        return metrics
    
    def _calculate_completeness(self, data: List[Dict[str, Any]]) -> float:
        """완전성 점수 계산"""
        if not data:
            return 0.0
        
        complete_count = 0
        total_fields = 0
        
        for item in data:
            content = item.get("content")
            if content:
                # 콘텐츠 존재 여부
                if isinstance(content, str) and len(content.strip()) > 0:
                    complete_count += 1
                elif isinstance(content, dict) and len(content) > 0:
                    complete_count += 1
                elif isinstance(content, list) and len(content) > 0:
                    complete_count += 1
            
            # 메타데이터 완전성
            metadata = item.get("metadata", {})
            required_metadata = ["generator_id", "timestamp"]
            for field in required_metadata:
                total_fields += 1
                if field in metadata:
                    complete_count += 1
        
        total_checks = len(data) + total_fields
        completeness = complete_count / total_checks if total_checks > 0 else 0.0
        
        return min(1.0, completeness)
    
    def _calculate_consistency(self, data: List[Dict[str, Any]]) -> float:
        """일관성 점수 계산"""
        if not data:
            return 0.0
        
        # 데이터 구조 일관성 체크
        structures = []
        for item in data:
            structure = self._get_data_structure(item)
            structures.append(structure)
        
        if not structures:
            return 0.0
        
        # 가장 일반적인 구조
        most_common = Counter(structures).most_common(1)[0]
        common_structure = most_common[0]
        common_count = most_common[1]
        
        # 일관성 비율
        consistency = common_count / len(structures)
        
        # 콘텐츠 형식 일관성
        content_types = set()
        for item in data:
            content = item.get("content")
            content_types.add(type(content).__name__)
        
        # 타입 다양성이 적을수록 일관성 높음
        type_consistency = 1.0 / len(content_types) if content_types else 0.0
        
        # 종합 일관성
        total_consistency = (consistency + type_consistency) / 2
        
        return min(1.0, total_consistency)
    
    def _calculate_uniqueness(self, data: List[Dict[str, Any]]) -> float:
        """고유성 점수 계산"""
        if not data:
            return 0.0
        
        contents = []
        for item in data:
            content = item.get("content", "")
            if isinstance(content, str):
                contents.append(content)
            else:
                contents.append(str(content))
        
        if not contents:
            return 0.0
        
        # 정확한 중복 체크
        unique_contents = set(contents)
        exact_uniqueness = len(unique_contents) / len(contents)
        
        # 유사도 기반 중복 체크
        if len(contents) > 1:
            similarity_scores = []
            for i in range(min(100, len(contents))):  # 샘플링
                for j in range(i + 1, min(100, len(contents))):
                    similarity = self._calculate_string_similarity(contents[i], contents[j])
                    similarity_scores.append(similarity)
            
            if similarity_scores:
                avg_similarity = np.mean(similarity_scores)
                similarity_uniqueness = 1.0 - avg_similarity
            else:
                similarity_uniqueness = 1.0
        else:
            similarity_uniqueness = 1.0
        
        # 종합 고유성
        uniqueness = (exact_uniqueness + similarity_uniqueness) / 2
        
        return min(1.0, uniqueness)
    
    def _calculate_validity(self, data: List[Dict[str, Any]], domain: str) -> float:
        """유효성 점수 계산"""
        if not data:
            return 0.0
        
        valid_count = 0
        
        for item in data:
            content = item.get("content", "")
            
            # 기본 유효성 체크
            if self._is_valid_content(content):
                valid_count += 1
                
                # 도메인별 추가 검증
                if domain == "medical":
                    if self._validate_medical_content(content):
                        valid_count += 0.5
                elif domain == "financial":
                    if self._validate_financial_content(content):
                        valid_count += 0.5
                elif domain == "legal":
                    if self._validate_legal_content(content):
                        valid_count += 0.5
        
        validity = valid_count / len(data) if data else 0.0
        
        return min(1.0, validity)
    
    async def _calculate_distribution_similarity(
        self,
        generated_data: List[Dict[str, Any]],
        reference_data: List[Dict[str, Any]]
    ) -> float:
        """분포 유사성 계산"""
        if not generated_data or not reference_data:
            return 0.0
        
        # 특징 추출
        gen_features = self._extract_features(generated_data)
        ref_features = self._extract_features(reference_data)
        
        if not gen_features or not ref_features:
            return 0.0
        
        similarities = []
        
        # 길이 분포 비교
        if "lengths" in gen_features and "lengths" in ref_features:
            ks_stat, p_value = stats.ks_2samp(gen_features["lengths"], ref_features["lengths"])
            length_similarity = p_value  # p-value가 높을수록 유사
            similarities.append(length_similarity)
        
        # 단어 빈도 분포 비교
        if "word_frequencies" in gen_features and "word_frequencies" in ref_features:
            word_similarity = self._compare_frequency_distributions(
                gen_features["word_frequencies"],
                ref_features["word_frequencies"]
            )
            similarities.append(word_similarity)
        
        # 구조적 특징 비교
        if "structural_features" in gen_features and "structural_features" in ref_features:
            struct_similarity = self._compare_structural_features(
                gen_features["structural_features"],
                ref_features["structural_features"]
            )
            similarities.append(struct_similarity)
        
        if similarities:
            return np.mean(similarities)
        
        return 0.5  # 기본값
    
    def _calculate_statistical_parity(
        self,
        generated_data: List[Dict[str, Any]],
        reference_data: List[Dict[str, Any]]
    ) -> float:
        """통계적 동등성 계산"""
        # 기본 통계량 비교
        gen_stats = self._calculate_basic_statistics(generated_data)
        ref_stats = self._calculate_basic_statistics(reference_data)
        
        if not gen_stats or not ref_stats:
            return 0.0
        
        differences = []
        
        for key in gen_stats:
            if key in ref_stats:
                if isinstance(gen_stats[key], (int, float)) and isinstance(ref_stats[key], (int, float)):
                    # 정규화된 차이 계산
                    if ref_stats[key] != 0:
                        diff = abs(gen_stats[key] - ref_stats[key]) / abs(ref_stats[key])
                    else:
                        diff = abs(gen_stats[key])
                    differences.append(min(1.0, diff))
        
        if differences:
            # 차이가 작을수록 점수 높음
            parity = 1.0 - np.mean(differences)
            return max(0.0, parity)
        
        return 0.5
    
    async def _calculate_text_quality_metrics(
        self,
        generated_data: List[Dict[str, Any]],
        reference_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """텍스트 품질 메트릭 계산"""
        metrics = {}
        
        # 텍스트 추출
        gen_texts = self._extract_texts(generated_data)
        
        if not gen_texts:
            return metrics
        
        # 가독성 점수
        metrics["readability"] = self._calculate_readability(gen_texts)
        
        # 어휘 다양성
        metrics["lexical_diversity"] = self._calculate_lexical_diversity(gen_texts)
        
        # 문법적 정확성 (간단한 휴리스틱)
        metrics["grammatical_correctness"] = self._calculate_grammatical_correctness(gen_texts)
        
        # 의미적 일관성
        metrics["semantic_coherence"] = await self._calculate_semantic_coherence(gen_texts)
        
        # 참조 데이터와 비교
        if reference_data:
            ref_texts = self._extract_texts(reference_data)
            if ref_texts:
                # 의미적 유사성
                metrics["semantic_similarity"] = await self._calculate_semantic_similarity(
                    gen_texts, ref_texts
                )
                
                # 스타일 유사성
                metrics["style_similarity"] = self._calculate_style_similarity(
                    gen_texts, ref_texts
                )
        
        return metrics
    
    def _calculate_readability(self, texts: List[str]) -> float:
        """가독성 점수 계산 (Flesch Reading Ease 변형)"""
        if not texts:
            return 0.0
        
        scores = []
        
        for text in texts[:100]:  # 샘플링
            sentences = text.split('.')
            words = text.split()
            
            if not sentences or not words:
                continue
            
            # 평균 문장 길이
            avg_sentence_length = len(words) / len(sentences)
            
            # 평균 음절 수 (간단한 추정)
            syllables = sum(self._count_syllables(word) for word in words)
            avg_syllables_per_word = syllables / len(words) if words else 0
            
            # Flesch Reading Ease 공식 변형
            score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
            
            # 0-1 범위로 정규화
            normalized_score = max(0, min(100, score)) / 100
            scores.append(normalized_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_lexical_diversity(self, texts: List[str]) -> float:
        """어휘 다양성 계산"""
        if not texts:
            return 0.0
        
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        # Type-Token Ratio
        unique_words = set(all_words)
        ttr = len(unique_words) / len(all_words)
        
        # 길이에 대한 보정
        if len(all_words) > 1000:
            # MTLD (Measure of Textual Lexical Diversity) 간단 버전
            chunk_size = 100
            chunk_ttrs = []
            
            for i in range(0, len(all_words), chunk_size):
                chunk = all_words[i:i+chunk_size]
                if len(chunk) > 10:
                    chunk_unique = set(chunk)
                    chunk_ttr = len(chunk_unique) / len(chunk)
                    chunk_ttrs.append(chunk_ttr)
            
            if chunk_ttrs:
                ttr = np.mean(chunk_ttrs)
        
        return min(1.0, ttr * 2)  # 스케일 조정
    
    def _calculate_grammatical_correctness(self, texts: List[str]) -> float:
        """문법적 정확성 계산 (휴리스틱)"""
        if not texts:
            return 0.0
        
        scores = []
        
        for text in texts[:100]:  # 샘플링
            score = 1.0
            
            # 기본 문법 규칙 체크
            # 1. 문장 시작 대문자
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and not sentence[0].isupper():
                    score *= 0.95
            
            # 2. 구두점 체크
            if not any(p in text for p in ['.', '!', '?']):
                score *= 0.9
            
            # 3. 괄호 짝 맞추기
            if text.count('(') != text.count(')'):
                score *= 0.9
            if text.count('[') != text.count(']'):
                score *= 0.9
            
            # 4. 인용부호 짝 맞추기
            if text.count('"') % 2 != 0:
                score *= 0.95
            
            # 5. 과도한 반복 체크
            words = text.lower().split()
            if len(words) > 3:
                for i in range(len(words) - 2):
                    if words[i] == words[i+1] == words[i+2]:
                        score *= 0.9
                        break
            
            scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    async def _calculate_semantic_coherence(self, texts: List[str]) -> float:
        """의미적 일관성 계산"""
        if not texts or len(texts) < 2:
            return 1.0
        
        try:
            # TF-IDF 벡터화
            sample_texts = texts[:min(100, len(texts))]
            
            if len(sample_texts) < 2:
                return 1.0
            
            vectors = self.vectorizer.fit_transform(sample_texts)
            
            # 문서 간 코사인 유사도 계산
            similarities = []
            for i in range(len(sample_texts) - 1):
                sim = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
                similarities.append(sim)
            
            # 평균 유사도 (너무 높거나 낮지 않은 것이 좋음)
            avg_similarity = np.mean(similarities)
            
            # 이상적인 범위: 0.3 ~ 0.7
            if 0.3 <= avg_similarity <= 0.7:
                coherence = 1.0
            elif avg_similarity < 0.3:
                coherence = avg_similarity / 0.3
            else:
                coherence = 1.0 - (avg_similarity - 0.7) / 0.3
            
            return max(0.0, min(1.0, coherence))
            
        except Exception as e:
            logger.warning(f"Semantic coherence calculation failed: {e}")
            return 0.5
    
    async def _calculate_semantic_similarity(
        self,
        gen_texts: List[str],
        ref_texts: List[str]
    ) -> float:
        """의미적 유사성 계산"""
        if not gen_texts or not ref_texts:
            return 0.0
        
        try:
            # 샘플링
            gen_sample = gen_texts[:min(50, len(gen_texts))]
            ref_sample = ref_texts[:min(50, len(ref_texts))]
            
            # 결합하여 벡터화
            all_texts = gen_sample + ref_sample
            vectors = self.vectorizer.fit_transform(all_texts)
            
            # 생성 데이터와 참조 데이터 간 평균 유사도
            gen_vectors = vectors[:len(gen_sample)]
            ref_vectors = vectors[len(gen_sample):]
            
            similarities = cosine_similarity(gen_vectors, ref_vectors)
            avg_similarity = np.mean(similarities)
            
            return min(1.0, avg_similarity)
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.5
    
    def _calculate_style_similarity(
        self,
        gen_texts: List[str],
        ref_texts: List[str]
    ) -> float:
        """스타일 유사성 계산"""
        gen_style = self._extract_style_features(gen_texts)
        ref_style = self._extract_style_features(ref_texts)
        
        if not gen_style or not ref_style:
            return 0.5
        
        similarities = []
        
        for feature in gen_style:
            if feature in ref_style:
                gen_val = gen_style[feature]
                ref_val = ref_style[feature]
                
                if ref_val != 0:
                    similarity = 1.0 - abs(gen_val - ref_val) / abs(ref_val)
                else:
                    similarity = 1.0 if gen_val == 0 else 0.0
                
                similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.5
    
    def _extract_style_features(self, texts: List[str]) -> Dict[str, float]:
        """스타일 특징 추출"""
        if not texts:
            return {}
        
        features = {}
        
        # 평균 문장 길이
        sentence_lengths = []
        for text in texts:
            sentences = text.split('.')
            for sentence in sentences:
                if sentence.strip():
                    sentence_lengths.append(len(sentence.split()))
        
        if sentence_lengths:
            features["avg_sentence_length"] = np.mean(sentence_lengths)
            features["std_sentence_length"] = np.std(sentence_lengths)
        
        # 평균 단어 길이
        word_lengths = []
        for text in texts:
            words = text.split()
            word_lengths.extend([len(word) for word in words])
        
        if word_lengths:
            features["avg_word_length"] = np.mean(word_lengths)
        
        # 구두점 사용 빈도
        punctuation_count = sum(text.count(p) for text in texts for p in '.,!?;:')
        total_chars = sum(len(text) for text in texts)
        if total_chars > 0:
            features["punctuation_ratio"] = punctuation_count / total_chars
        
        # 대문자 비율
        upper_count = sum(1 for text in texts for char in text if char.isupper())
        if total_chars > 0:
            features["uppercase_ratio"] = upper_count / total_chars
        
        return features
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """종합 품질 점수 계산"""
        # 가중치 설정
        weights = {
            "completeness": 0.15,
            "consistency": 0.15,
            "uniqueness": 0.20,
            "validity": 0.20,
            "distribution_similarity": 0.10,
            "statistical_parity": 0.10,
            "readability": 0.05,
            "semantic_coherence": 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score = metrics[metric]
                if isinstance(score, (int, float)):
                    total_score += score * weight
                    total_weight += weight
        
        if total_weight > 0:
            overall = total_score / total_weight
        else:
            # 폴백: 사용 가능한 메트릭의 평균
            available_scores = [v for v in metrics.values() if isinstance(v, (int, float))]
            overall = np.mean(available_scores) if available_scores else 0.0
        
        return min(1.0, max(0.0, overall))
    
    # 유틸리티 메서드들
    def _get_data_structure(self, item: Dict[str, Any]) -> str:
        """데이터 구조 시그니처 생성"""
        structure = []
        for key in sorted(item.keys()):
            value = item[key]
            structure.append(f"{key}:{type(value).__name__}")
        return "|".join(structure)
    
    def _calculate_string_similarity(self, s1: str, s2: str) -> float:
        """문자열 유사도 계산"""
        if s1 == s2:
            return 1.0
        
        # 간단한 Jaccard 유사도
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _is_valid_content(self, content: Any) -> bool:
        """콘텐츠 유효성 체크"""
        if content is None:
            return False
        
        if isinstance(content, str):
            return len(content.strip()) >= 10
        elif isinstance(content, dict):
            return len(content) > 0
        elif isinstance(content, list):
            return len(content) > 0
        
        return True
    
    def _validate_medical_content(self, content: str) -> bool:
        """의료 콘텐츠 검증"""
        if not isinstance(content, str):
            return False
        
        # 의료 키워드 체크
        medical_keywords = [
            "patient", "diagnosis", "treatment", "symptom", "medical",
            "doctor", "hospital", "medicine", "health", "clinical"
        ]
        
        content_lower = content.lower()
        keyword_found = any(keyword in content_lower for keyword in medical_keywords)
        
        # PHI 패턴이 없는지 체크
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10,}\b',  # 긴 숫자 (의료 기록 번호 등)
        ]
        
        for pattern in phi_patterns:
            if re.search(pattern, content):
                return False  # PHI 포함 시 유효하지 않음
        
        return keyword_found
    
    def _validate_financial_content(self, content: str) -> bool:
        """금융 콘텐츠 검증"""
        if not isinstance(content, str):
            return False
        
        # 금융 키워드 체크
        financial_keywords = [
            "investment", "market", "stock", "bond", "portfolio",
            "return", "risk", "asset", "finance", "trading"
        ]
        
        content_lower = content.lower()
        keyword_found = any(keyword in content_lower for keyword in financial_keywords)
        
        # 민감 정보 패턴 체크
        sensitive_patterns = [
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # 신용카드
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, content):
                return False
        
        return keyword_found
    
    def _validate_legal_content(self, content: str) -> bool:
        """법률 콘텐츠 검증"""
        if not isinstance(content, str):
            return False
        
        # 법률 키워드 체크
        legal_keywords = [
            "legal", "law", "court", "attorney", "contract",
            "agreement", "clause", "litigation", "compliance", "regulation"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in legal_keywords)
    
    def _is_text_data(self, data: List[Dict[str, Any]]) -> bool:
        """텍스트 데이터 여부 확인"""
        if not data:
            return False
        
        # 샘플 체크
        sample = data[:10] if len(data) > 10 else data
        text_count = sum(1 for item in sample if isinstance(item.get("content"), str))
        
        return text_count > len(sample) / 2
    
    def _extract_texts(self, data: List[Dict[str, Any]]) -> List[str]:
        """텍스트 추출"""
        texts = []
        for item in data:
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
        return texts
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """특징 추출"""
        features = {}
        
        # 텍스트 길이
        lengths = []
        for item in data:
            content = item.get("content", "")
            if isinstance(content, str):
                lengths.append(len(content.split()))
        
        if lengths:
            features["lengths"] = lengths
        
        # 단어 빈도
        word_freq = Counter()
        for item in data:
            content = item.get("content", "")
            if isinstance(content, str):
                words = content.lower().split()
                word_freq.update(words)
        
        if word_freq:
            features["word_frequencies"] = dict(word_freq.most_common(100))
        
        # 구조적 특징
        structures = []
        for item in data:
            structure = self._get_data_structure(item)
            structures.append(structure)
        
        if structures:
            features["structural_features"] = Counter(structures)
        
        return features
    
    def _compare_frequency_distributions(
        self,
        freq1: Dict[str, int],
        freq2: Dict[str, int]
    ) -> float:
        """빈도 분포 비교"""
        if not freq1 or not freq2:
            return 0.0
        
        # 공통 키 찾기
        common_keys = set(freq1.keys()).intersection(set(freq2.keys()))
        
        if not common_keys:
            return 0.0
        
        # 정규화
        total1 = sum(freq1.values())
        total2 = sum(freq2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        # KL divergence 계산 (간단한 버전)
        divergence = 0.0
        for key in common_keys:
            p = freq1[key] / total1
            q = freq2[key] / total2
            if p > 0 and q > 0:
                divergence += p * np.log(p / q)
        
        # 유사도로 변환 (0~1)
        similarity = np.exp(-divergence)
        
        return min(1.0, similarity)
    
    def _compare_structural_features(
        self,
        struct1: Counter,
        struct2: Counter
    ) -> float:
        """구조적 특징 비교"""
        if not struct1 or not struct2:
            return 0.0
        
        # Jaccard 유사도
        keys1 = set(struct1.keys())
        keys2 = set(struct2.keys())
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_basic_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
       """기본 통계량 계산"""
       stats = {}
       
       if not data:
           return stats
       
       # 텍스트 길이 통계
       lengths = []
       for item in data:
           content = item.get("content", "")
           if isinstance(content, str):
               lengths.append(len(content.split()))
       
       if lengths:
           stats["mean_length"] = np.mean(lengths)
           stats["std_length"] = np.std(lengths)
           stats["min_length"] = np.min(lengths)
           stats["max_length"] = np.max(lengths)
           stats["median_length"] = np.median(lengths)
       
       # 샘플 수
       stats["sample_count"] = len(data)
       
       # 고유값 비율
       unique_contents = set()
       for item in data:
           content = str(item.get("content", ""))
           unique_contents.add(content)
       
       stats["unique_ratio"] = len(unique_contents) / len(data) if data else 0
       
       return stats
   
    def _count_syllables(self, word: str) -> int:
        """단어의 음절 수 추정"""
        word = word.lower()
        vowels = "aeiou"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # 최소 1음절
        return max(1, syllable_count)
    
    def _save_metrics_history(self, metrics: Dict[str, float]):
        """메트릭 히스토리 저장"""
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy()
        })
        
        # 히스토리 크기 제한
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """메트릭 요약 정보 반환"""
        if not self.metrics_history:
            return {}
        
        # 최근 메트릭
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        summary = {
            "total_evaluations": len(self.metrics_history),
            "recent_average_quality": np.mean([
                m["metrics"].get("overall_quality_score", 0) 
                for m in recent_metrics
            ]),
            "quality_trend": self._calculate_trend([
                m["metrics"].get("overall_quality_score", 0) 
                for m in self.metrics_history
            ])
        }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """트렌드 계산"""
        if len(values) < 2:
            return "stable"
        
        # 선형 회귀로 트렌드 계산
        x = np.arange(len(values))
        y = np.array(values)
        
        # 간단한 선형 회귀
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"_