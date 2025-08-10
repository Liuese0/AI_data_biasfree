# 편향성 제거 합성 데이터 생성 플랫폼

AI 학습용 고품질 합성 데이터를 생성하는 플랫폼입니다. 편향성을 15-25% 감소시키면서 데이터 품질 90-95%를 유지합니다.

## 주요 기능

### 🚀 핵심 기능
- **다중 생성기 앙상블 시스템**: 5-10개 생성기를 활용한 앙상블 아키텍처
- **편향성 완화**: 인구통계학적 균형, 성별/연령/문화 다양성 보장
- **실시간 품질 검증**: 5개 품질 지표 실시간 모니터링
- **도메인 특화 생성**: 의료, 금융, 법률 도메인별 최적화

### 📊 성과 지표
- 편향성 15-25% 감소
- 데이터 품질 90-95% 유지
- 재식별 위험 5% 미만
- k-익명성(k≥5) 보장

## 시작하기

### 사전 요구사항
- Docker & Docker Compose
- Node.js 18+ (개발용)
- Python 3.11+ (개발용)
- CUDA 지원 GPU (권장)

### 설치 및 실행

#### 1. Docker를 사용한 실행 (권장)

```bash
# 저장소 클론
git clone https://github.com/your-org/bias-free-synthetic-data.git
cd bias-free-synthetic-data

# 환경 변수 설정
cp backend/.env.example backend/.env
# .env 파일을 편집하여 필요한 값 설정

# Docker Compose로 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f