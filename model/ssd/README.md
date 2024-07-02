graph TD
    A[학습 시작] --> B[입력 이미지 로드 및 전처리]
    B --> C[SSD 모델 순전파]
    C --> D[특징 추출]
    D --> E[VGG16 통과]
    E --> F[추가 컨볼루션 레이어 통과]
    
    F --> G1[Feature Map 1 생성]
    F --> G2[Feature Map 2 생성]
    F --> G3[Feature Map 3 생성]
    F --> G4[Feature Map 4 생성]
    F --> G5[Feature Map 5 생성]
    F --> G6[Feature Map 6 생성]
    
    G1 --> H[위치 및 클래스 예측]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H
    G6 --> H
    
    H --> I[위치 예측 레이어]
    H --> J[분류 예측 레이어]
    I --> K[위치 손실 계산]
    J --> L[클래스 손실 계산]
    K --> M[IoU 계산]
    L --> M
    M --> N[포지티브 및 네거티브 샘플 구분]
    N --> O[Hard Negative Mining]
    O --> P[총 손실 계산]
    P --> Q[손실 역전파 및 가중치 업데이트]
    Q --> R[다음 배치 학습]
    R --> S[학습 종료]
    N --> T[NMS 적용]
    T --> U[최종 박스 및 클래스 선택]
