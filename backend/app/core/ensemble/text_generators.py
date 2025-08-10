import asyncio
import random
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import logging
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BertForMaskedLM, BertTokenizer,
    AutoModelForCausalLM, AutoTokenizer
)

from app.core.ensemble.base_generator import BaseGenerator, TransformerTextGenerator
from app.config import settings

logger = logging.getLogger(__name__)

class GPT2Generator(TransformerTextGenerator):
    """GPT-2 기반 생성기"""
    
    def __init__(self, generator_id: str, variant: str = "gpt2"):
        super().__init__(generator_id, variant)
        self.specialization = "conversational"
        
    async def generate(
        self,
        prompt: str,
        quantity: int,
        constraints: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """GPT-2 특화 생성"""
        constraints = constraints or {}
        
        # GPT-2 특화 파라미터
        constraints.update({
            "temperature": constraints.get("temperature", 0.8),
            "top_p": constraints.get("top_p", 0.9),
            "repetition_penalty": 1.2,
            "max_length": constraints.get("max_length", 150)
        })
        
        # 대화형 데이터 생성을 위한 프롬프트 조정
        if constraints.get("dialogue_mode"):
            prompt = self._format_dialogue_prompt(prompt)
        
        return await super().generate(prompt, quantity, constraints)
    
    def _format_dialogue_prompt(self, prompt: str) -> str:
        """대화형 프롬프트 포맷팅"""
        dialogue_starters = [
            "Human: {}\nAssistant:",
            "Q: {}\nA:",
            "User: {}\nSystem:",
            "[대화 시작]\n사용자: {}\n응답:"
        ]
        
        starter = random.choice(dialogue_starters)
        return starter.format(prompt)

class T5Generator(BaseGenerator):
    """T5 기반 생성기"""
    
    def __init__(self, generator_id: str, variant: str = "t5-small"):
        super().__init__(generator_id, variant, "text")
        self.tokenizer = None
        self.model = None
        self.specialization = "structured"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """T5 모델 초기화"""
        try:
            logger.info(f"Initializing T5 model: {self.model_name}")
            
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            self.is_initialized = True
            logger.info(f"T5 model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize T5: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        quantity: int,
        constraints: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """T5 기반 텍스트 생성"""
        if not self.is_initialized:
            await self.initialize()
        
        generated_data = []
        constraints = constraints or {}
        
        # T5 특화 태스크 프리픽스
        task_prefix = self._get_task_prefix(constraints.get("task_type", "generate"))
        input_text = f"{task_prefix}: {prompt}"
        
        try:
            start_time = datetime.now()
            
            for i in range(quantity):
                # 입력 토크나이징
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # 생성
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=constraints.get("max_length", 150),
                        min_length=constraints.get("min_length", 10),
                        temperature=constraints.get("temperature", 0.7),
                        do_sample=True,
                        top_p=constraints.get("top_p", 0.9),
                        num_beams=constraints.get("num_beams", 4)
                    )
                
                # 디코딩
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # 구조화된 데이터 생성
                if constraints.get("structured_output"):
                    generated_text = self._structure_output(generated_text, constraints)
                
                generated_data.append({
                    "content": generated_text,
                    "metadata": {
                        "generator_id": self.generator_id,
                        "model": self.model_name,
                        "task_type": constraints.get("task_type", "generate"),
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # 배치 처리를 위한 짧은 대기
                if i % 10 == 0:
                    await asyncio.sleep(0.1)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"T5 generated {len(generated_data)} samples in {generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"T5 generation failed: {e}")
            raise
        
        return generated_data
    
    async def validate_output(
        self,
        generated_data: List[Dict[str, Any]]
    ) -> tuple[bool, float]:
        """T5 출력 검증"""
        if not generated_data:
            return False, 0.0
        
        valid_count = 0
        quality_scores = []
        
        for sample in generated_data:
            content = sample.get("content", "")
            
            # 구조 검증
            if self._validate_structure(content):
                valid_count += 1
                quality_scores.append(0.9)
            else:
                quality_scores.append(0.3)
        
        validity_ratio = valid_count / len(generated_data)
        avg_quality = np.mean(quality_scores)
        
        return validity_ratio >= 0.8, avg_quality
    
    def _get_task_prefix(self, task_type: str) -> str:
        """T5 태스크 프리픽스"""
        prefixes = {
            "generate": "generate text",
            "summarize": "summarize",
            "translate": "translate English to Korean",
            "question": "generate question",
            "paraphrase": "paraphrase"
        }
        return prefixes.get(task_type, "generate text")
    
    def _structure_output(self, text: str, constraints: Dict[str, Any]) -> str:
        """출력 구조화"""
        structure_type = constraints.get("structure_type", "plain")
        
        if structure_type == "json":
            # JSON 형식으로 변환
            import json
            try:
                data = {
                    "text": text,
                    "type": constraints.get("data_type", "general"),
                    "generated_at": datetime.now().isoformat()
                }
                return json.dumps(data, ensure_ascii=False)
            except:
                return text
        
        elif structure_type == "dialogue":
            # 대화 형식으로 변환
            lines = text.split(".")
            dialogue = []
            for i, line in enumerate(lines):
                if line.strip():
                    speaker = "A" if i % 2 == 0 else "B"
                    dialogue.append(f"{speaker}: {line.strip()}")
            return "\n".join(dialogue)
        
        return text
    
    def _validate_structure(self, text: str) -> bool:
        """구조 유효성 검증"""
        if not text or len(text) < 10:
            return False
        
        # JSON 형식 체크
        if text.startswith("{") and text.endswith("}"):
            try:
                import json
                json.loads(text)
                return True
            except:
                pass
        
        return True

class BERTGenerator(BaseGenerator):
    """BERT 기반 마스킹 언어 모델 생성기"""
    
    def __init__(self, generator_id: str, variant: str = "bert-base-uncased"):
        super().__init__(generator_id, variant, "text")
        self.tokenizer = None
        self.model = None
        self.specialization = "fill-mask"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """BERT 모델 초기화"""
        try:
            logger.info(f"Initializing BERT model: {self.model_name}")
            
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            
            self.model = BertForMaskedLM.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            self.is_initialized = True
            logger.info("BERT model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        quantity: int,
        constraints: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """BERT 기반 텍스트 생성 (마스크 채우기)"""
        if not self.is_initialized:
            await self.initialize()
        
        generated_data = []
        constraints = constraints or {}
        
        try:
            # 템플릿 기반 생성
            templates = self._get_templates(constraints.get("domain", "general"))
            
            for i in range(quantity):
                template = random.choice(templates)
                
                # 마스크 토큰 추가
                masked_text = self._add_masks_to_template(template, prompt)
                
                # 토크나이징
                inputs = self.tokenizer(
                    masked_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # 마스크 위치 찾기
                mask_token_index = torch.where(
                    inputs["input_ids"] == self.tokenizer.mask_token_id
                )[1]
                
                # 예측
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = outputs.logits
                
                # 각 마스크 위치에 대해 토큰 선택
                generated_text = masked_text
                for mask_idx in mask_token_index:
                    mask_logits = predictions[0, mask_idx]
                    
                    # Top-k 샘플링
                    top_k = constraints.get("top_k", 10)
                    top_k_tokens = torch.topk(mask_logits, top_k).indices
                    
                    # 랜덤 선택 (다양성을 위해)
                    selected_token = random.choice(top_k_tokens.tolist())
                    token_str = self.tokenizer.decode([selected_token])
                    
                    # 첫 번째 마스크만 교체 (순차적 처리)
                    generated_text = generated_text.replace(
                        self.tokenizer.mask_token, 
                        token_str, 
                        1
                    )
                
                # 정리
                generated_text = self._clean_bert_output(generated_text)
                
                generated_data.append({
                    "content": generated_text,
                    "metadata": {
                        "generator_id": self.generator_id,
                        "model": self.model_name,
                        "method": "mask-filling",
                        "template_used": template,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                if i % 10 == 0:
                    await asyncio.sleep(0.1)
            
            logger.info(f"BERT generated {len(generated_data)} samples")
            
        except Exception as e:
            logger.error(f"BERT generation failed: {e}")
            raise
        
        return generated_data
    
    async def validate_output(
        self,
        generated_data: List[Dict[str, Any]]
    ) -> tuple[bool, float]:
        """BERT 출력 검증"""
        if not generated_data:
            return False, 0.0
        
        valid_count = 0
        for sample in generated_data:
            content = sample.get("content", "")
            
            # 마스크 토큰이 남아있는지 체크
            if self.tokenizer.mask_token not in content and len(content) > 10:
                valid_count += 1
        
        validity_ratio = valid_count / len(generated_data)
        return validity_ratio >= 0.8, validity_ratio
    
    def _get_templates(self, domain: str) -> List[str]:
        """도메인별 템플릿"""
        templates = {
            "medical": [
                "The patient presented with [MASK] and was diagnosed with [MASK].",
                "Treatment for [MASK] typically includes [MASK] and [MASK].",
                "The [MASK] symptoms indicate a possible [MASK] condition.",
            ],
            "financial": [
                "The [MASK] market showed [MASK] growth in [MASK].",
                "Investment in [MASK] yielded [MASK] returns over [MASK].",
                "The [MASK] index closed at [MASK] points, a [MASK] change.",
            ],
            "general": [
                "The [MASK] is [MASK] and [MASK].",
                "[MASK] announced that [MASK] will [MASK].",
                "According to [MASK], the [MASK] has [MASK] significantly.",
            ]
        }
        
        return templates.get(domain, templates["general"])
    
    def _add_masks_to_template(self, template: str, prompt: str) -> str:
        """템플릿에 마스크 추가"""
        # 프롬프트의 키워드를 추출하여 일부 마스크 대체
        keywords = prompt.split()[:3]  # 처음 3개 단어 사용
        
        masked = template
        for keyword in keywords:
            if "[MASK]" in masked:
                # 50% 확률로 키워드로 대체
                if random.random() > 0.5:
                    masked = masked.replace("[MASK]", keyword, 1)
        
        # 남은 [MASK]를 BERT 마스크 토큰으로 변경
        masked = masked.replace("[MASK]", self.tokenizer.mask_token)
        
        return masked
    
    def _clean_bert_output(self, text: str) -> str:
        """BERT 출력 정리"""
        # 특수 토큰 제거
        text = text.replace(self.tokenizer.pad_token or "", "")
        text = text.replace(self.tokenizer.sep_token or "", "")
        text = text.replace(self.tokenizer.cls_token or "", "")
        
        # 공백 정리
        text = " ".join(text.split())
        
        return text.strip()

class DiversityGenerator(BaseGenerator):
    """다양성 강화 생성기"""
    
    def __init__(self, generator_id: str):
        super().__init__(generator_id, "diversity-enhancer", "text")
        self.base_generators = []
        self.specialization = "diversity"
        
    async def initialize(self):
        """다양성 생성기 초기화"""
        self.is_initialized = True
        logger.info("Diversity generator initialized")
    
    async def generate(
        self,
        prompt: str,
        quantity: int,
        constraints: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """다양성 강화 생성"""
        generated_data = []
        constraints = constraints or {}
        
        # 다양한 변형 생성
        variations = self._create_prompt_variations(prompt, constraints)
        
        samples_per_variation = max(1, quantity // len(variations))
        
        for variation in variations:
            for _ in range(samples_per_variation):
                if len(generated_data) >= quantity:
                    break
                
                # 변형된 텍스트 생성
                text = self._generate_diverse_text(variation, constraints)
                
                generated_data.append({
                    "content": text,
                    "metadata": {
                        "generator_id": self.generator_id,
                        "variation_type": variation["type"],
                        "diversity_score": self._calculate_diversity_score(text),
                        "timestamp": datetime.now().isoformat()
                    }
                })
        
        return generated_data
    
    async def validate_output(
        self,
        generated_data: List[Dict[str, Any]]
    ) -> tuple[bool, float]:
        """다양성 검증"""
        if not generated_data:
            return False, 0.0
        
        # 다양성 점수 계산
        texts = [item["content"] for item in generated_data]
        diversity_score = self._calculate_set_diversity(texts)
        
        return diversity_score >= 0.7, diversity_score
    
    def _create_prompt_variations(
        self, 
        prompt: str, 
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """프롬프트 변형 생성"""
        variations = []
        
        # 관점 변형
        perspectives = ["first-person", "third-person", "objective", "subjective"]
        for perspective in perspectives:
            variations.append({
                "type": perspective,
                "prompt": self._modify_perspective(prompt, perspective)
            })
        
        # 스타일 변형
        styles = ["formal", "casual", "technical", "creative"]
        for style in styles:
            variations.append({
                "type": style,
                "prompt": self._modify_style(prompt, style)
            })
        
        # 문화적 변형
        if constraints.get("bias_mitigation", {}).get("cultural_diversity"):
            cultures = ["western", "eastern", "african", "latin"]
            for culture in cultures:
                variations.append({
                    "type": f"cultural_{culture}",
                    "prompt": self._add_cultural_context(prompt, culture)
                })
        
        return variations[:10]  # 최대 10개 변형
    
    def _generate_diverse_text(
        self, 
        variation: Dict[str, Any], 
        constraints: Dict[str, Any]
    ) -> str:
        """다양한 텍스트 생성"""
        base_text = variation["prompt"]
        variation_type = variation["type"]
        
        # 템플릿 기반 생성
        templates = {
            "first-person": "I believe that {}",
            "third-person": "They observed that {}",
            "objective": "The data shows that {}",
            "subjective": "It seems that {}",
            "formal": "It is hereby stated that {}",
            "casual": "So basically, {}",
            "technical": "The analysis indicates that {}",
            "creative": "Imagine if {}"
        }
        
        template = templates.get(variation_type, "{}")
        generated = template.format(base_text)
        
        # 추가 다양성 요소
        if random.random() > 0.5:
            generated = self._add_random_elements(generated)
        
        return generated
    
    def _modify_perspective(self, text: str, perspective: str) -> str:
        """관점 수정"""
        if perspective == "first-person":
            text = text.replace("one might", "I might")
            text = text.replace("people", "we")
        elif perspective == "third-person":
            text = text.replace("I", "they")
            text = text.replace("we", "they")
        return text
    
    def _modify_style(self, text: str, style: str) -> str:
        """스타일 수정"""
        if style == "formal":
            replacements = {
                "don't": "do not",
                "can't": "cannot",
                "won't": "will not"
            }
        elif style == "casual":
            replacements = {
                "do not": "don't",
                "cannot": "can't",
                "will not": "won't"
            }
        else:
            return text
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _add_cultural_context(self, text: str, culture: str) -> str:
        """문화적 맥락 추가"""
        contexts = {
            "western": "In Western perspective, ",
            "eastern": "From an Eastern viewpoint, ",
            "african": "In African context, ",
            "latin": "From Latin American perspective, "
        }
        return contexts.get(culture, "") + text
    
    def _add_random_elements(self, text: str) -> str:
        """무작위 요소 추가"""
        elements = [
            " Additionally,",
            " Furthermore,",
            " However,",
            " Meanwhile,",
            " Interestingly,"
        ]
        
        element = random.choice(elements)
        sentences = text.split(".")
        
        if len(sentences) > 1:
            insert_pos = random.randint(1, len(sentences)-1)
            sentences[insert_pos] = element + sentences[insert_pos]
            text = ".".join(sentences)
        
        return text
    
    def _calculate_diversity_score(self, text: str) -> float:
        """개별 텍스트 다양성 점수"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        
        # 문장 길이 다양성
        sentences = text.split(".")
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                length_variance = np.std(lengths) / (np.mean(lengths) + 1)
                diversity = (diversity + length_variance) / 2
        
        return min(1.0, diversity)
    
    def _calculate_set_diversity(self, texts: List[str]) -> float:
        """텍스트 집합 다양성 점수"""
        if not texts:
            return 0.0
        
        # 모든 텍스트의 n-gram 수집
        all_ngrams = set()
        total_ngrams = 0
        
        for text in texts:
            words = text.lower().split()
            # 2-gram 생성
            for i in range(len(words) - 1):
                ngram = f"{words[i]} {words[i+1]}"
                all_ngrams.add(ngram)
                total_ngrams += 1
        
        if total_ngrams == 0:
            return 0.0
        
        # 고유 n-gram 비율
        diversity = len(all_ngrams) / total_ngrams
        
        return min(1.0, diversity * 2)  # 스케일 조정