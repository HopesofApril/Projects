# Class Imbalance Framework

다양한 도메인 데이터에 존재하는 클래스 불균형 문제를 해결하기 위한 프로젝트입니다.  
문제 해결 전략의 일관성을 정리하고 여러 실험 코드를 하나의 공통 프레임워크 관점에서 재구성하였습니다.

## 💡 Core Idea

- 소수 클래스 성능 개선 중심 설계
- Data-level과 Algorithm-level을 결합한 Hybrid 전략
- 재현 가능한 실험 파이프라인 구성
- 도메인에 독립적인 구조

## 📝 General Pipeline

1. Data Preprocessing  
2. Imbalance Handling (Data-level)  
3. Model Design & Training  
4. Evaluation (Precision, Recall, F1-score, AUC 중심 평가)

각 도메인별 실험은 위 구조를 기반으로 구성되어 있습니다.

## 🗂️ Applied Domains

- Network Intrusion Detection (SCI Journal, 2025)
- Financial Telemarketing
- Hazardous NEO Prediction
- Breast Cancer Dataset (Conference Best Paper, 2026)
- Additional Imbalanced Datasets
