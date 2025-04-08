# Introduction

1. 쌍선형 쌍(Bilinear Pairing)이란?

-   쌍선형 맵(Bilinear Map) 또는 **쌍선형 쌍(Bilinear Pairing)**은 $e: G_1 \times G_2 \rightarrow G_T$ 형태의 연산으로, 군(예: 타원곡선 군) 위에서 정의되는 특수한 함수입니다.
-   쌍선형성(Bilinearity): $e(aP, bQ) = e(P, Q)^{ab}$ 처럼, 스칼라 곱(sc. multiplication)과 쌍선형 맵이 서로 호환되는 성질을 가집니다.

2. “a, b, c” 예시와 암호화

-   예를 들어, $a, b, c$ 가 있을 때, $ab = c$ 라고 해봅시다.
-   여기서 이를 암호화한 형태는 $E(a), E(b), E(c)$ 가 됩니다. 각각 “쌍선형 쌍 위”에서 암호화(혹은 서명)된 상태라고 할 수 있습니다.
-   쌍선형 연산을 쓰면, 검증자는 실제 값을 알지 못하더라도 $E(a)E(b) = E(c)$ 를 확인할 수 있습니다.

# Prerequisites

-   타원곡선의 점 덧셈과 스칼라 곱을 이해해야합니다.
-

# How bilinear pairings work

## Notation

## Generalization, checking if two products are equal

## What “bilinear” means

# What is $e(P, Q)$ returning?

# Symmetric and Asymmetric Groups

# Field Extensions and the `G2` point in Python

## The G2 point in Python

# Bilinear Pairings in Python

## Equality of products

## The binary operator of $G\_{T}$

# Bilinear Pairings in Ethereum

## EIP 197 Specification

### Justification for EIP 197 design decision

### Sum of discrete logarithms

# End to End Solidity Example of Bilinear Pairings
