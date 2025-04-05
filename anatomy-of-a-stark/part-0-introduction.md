# What Are STARKs?

STARK는 “Scalable Transparent ARguments of Knowledge”의 약자로, 영지식 증명(ZK Proof) 시스템의 한 종류입니다. 기존의 zk-SNARKs와 유사한 맥락에 있지만, **투명성(transparent)**을 가지며, 보통 트러스티드 셋업(trusted setup) 없이도 동작한다는 장점이 있습니다. 또한 양자 컴퓨팅(quantum) 환경에서도 안전성을 유지할 수 있다고 여겨져, 최근 블록체인 등에서 주목받고 있습니다.

## zk-SNARKs 의 특징

-   (보편성, universal): 임의의 계산에 대해 무결성을 증명 가능
-   (비대화성, non-interactive): 증명 과정이 상호작용(interactive) 없이도 검증 가능
-   (효율적 검증, efficiently verifiable): 검증자가 전체 계산을 직접 재현하지 않고도 무결성을 빠르게 확인 가능
-   (영지식, zero-knowledge): 증명을 통해 계산 과정의 세부정보(비밀)는 노출되지 않음

## zk-SNARKs와의 비교

-   zk-SNARKs 역시 영지식 증명 방식이지만, 보통 특정한 초기 설정(“트러스트드 셋업”)이 필요합니다.
-   STARKs는 이 과정 없이도 대규모 계산의 무결성을 증명할 수 있어 투명성이 높습니다.

### 1. 암호학적 가정(Cryptographic Assumption)

-   기존 zk-SNARKs:

    -   이중 선형 쌍(Bilinear Pairing) 등을 사용하는 고급 암호학적 난제에 의존함.
    -   이 때문에 양자(Quantum) 환경에서의 안전성이나 특정 가정(“unfalsifiable assumptions”)에 대한 우려가 남음.

-   STARKs:
    -   충돌 회피 해시 함수(Collision-Resistant Hash Function) 하나에만 의존.
    -   이 해시 함수가 이상적으로 안전하다고 가정하면, 양자 컴퓨팅 환경에서도 안전하다고 “증명”할 수 있음(“provably post-quantum under an idealized model”).

### 2. Arithmetization(산술화)와 성능 최적화

-   전통 SNARKs:

    -   특정 암호학적 구조(예: bilinear map)에 맞춘 폴리노미얼(다항식) 표현 방식을 사용해야 함.
    -   이로 인해 성능 최적화에 제약이 있을 수 있음.

-   STARKs:
    -   “어떤 암호학적 난제를 쓰는지”와 독립적으로 산술화(Arithmetization) 필드를 자유롭게 선택할 수 있음.
    -   성능(속도, 증명 크기 등)을 최적화하기에 유리.

### 3. Trusted Setup(신뢰 설정) 여부

-   기존 zk-SNARKs:

    -   초기 설정 단계(“TrustedSetup Ceremony”)가 필요함.
    -   이 과정에서 생성된 비공개 매개변수(‘cryptographic toxic waste’)를 완전히 폐기해야만 안전성이 유지됨.
    -   만약 누군가가 이 폐기물(비밀 정보를) 보관하고 있으면, 임의로 위조된 증명을 만들 수 있게 됨.

-   STARKs:
    -   Trusted Setup이 필요 없음 → “투명성(transparent)”을 달성.
    -   당연히 ‘폐기물(toxic waste)’ 같은 것도 생기지 않으므로, 초기 설정에 대한 신뢰 문제나 폐기 절차가 필요 없다.

# Why?

-   기존 튜토리얼은 피상적: 개념을 높은 수준에서만 설명하고 있어, 실제 내부 동작 방식이나 수학적 근거를 깊이 이해하기 어렵다.
-   학술 논문은 난해: 논문들은 매우 전문적이고 방대하여, 일반 개발자나 초심자가 쉽게 접근하기 힘들다.
-   중간 단계를 채워 줄 자료 부족: 실제 구현과 수학 이론 사이를 연결해 주는 자료가 적다.

# Required Background Knowledge

미리 알아두면 좋은 주제

-   유한체(Finite fields)와 그 확장체(Extension fields)
-   유한체 위의 다항식(Polynomials) — 일변수(univariate)와 다변수(multivariate) 모두
-   고속 푸리에 변환(FFT, Fast Fourier Transform)
-   해시 함수(Hash functions)
