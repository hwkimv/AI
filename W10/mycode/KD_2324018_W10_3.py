# -*- coding: utf-8 -*-

import numpy as np
import mnist_loader as mnist

# 유틸 함수
def one_hot(labels, num_class=10):
    """정답 숫자(0~9)를 one-hot(예: 3 → [0,0,0,1,0,0,0,0,0,0]) 형태로 바꿈"""
    oh = np.zeros((len(labels), num_class), dtype=np.float32)
    for i in range(len(labels)):
        oh[i, int(labels[i])] = 1.0
    return oh

def sigmoid(z):
    """시그모이드 함수: 값(점수)을 0~1 사이로 압축"""
    return 1.0 / (1.0 + np.exp(-z))

def accuracy(y_true_int, y_pred_int):
    """정답/예측(정수라벨) 비교해서 정확도"""
    return np.mean((y_true_int == y_pred_int).astype(np.float32))



# 데이터 로드 & 전처리

# (x_train, y_train): x는 (N, 28, 28) uint8, y는 (N,) int
# (x_test,  y_test) : 같은 형식

# MNIST = 0~9 손글씨 숫자 이미지 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 학습/테스트에서 각각 1000개씩만 사용
N_TRAIN = 1000
N_TEST  = 10000
x_train = x_train[:N_TRAIN]
y_train = y_train[:N_TRAIN]
x_test  = x_test[:N_TEST]
y_test  = y_test[:N_TEST]

# (28x28 픽셀) 이미지를 (784)개의 숫자열로 바꾸기
# 각 픽셀 값(0~255)을 0~1 사이로 바꿔줌 (밝기 비율)
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test  = x_test.reshape(x_test.shape[0],  -1).astype(np.float32) / 255.0

# 각 이미지 앞에 1을 추가 (bias = 기준값 역할)
# bias 항을 위해 맨 앞에 1 붙이기 (입력 차원: 784 → 785)
x_train = np.hstack((np.ones((x_train.shape[0], 1), dtype=np.float32), x_train))
x_test  = np.hstack((np.ones((x_test.shape[0],  1), dtype=np.float32), x_test))

# 정수 라벨 보관(정확도/오류율 계산용)
# 정답 숫자 보관 (정확도 계산용)
y_train_int = y_train.astype(np.int32)
y_test_int  = y_test.astype(np.int32)

# one-hot 라벨(학습용)
y_train = one_hot(y_train_int, 10)
y_test  = one_hot(y_test_int,  10)

print("입력 특징 shape(학습/테스트):", x_train.shape, x_test.shape)
print("레이블 shape(학습/테스트):   ", y_train.shape, y_test.shape)
# 예: (1000, 785) (1000, 785) / (1000, 10) (1000, 10)


# 퍼셉트론 10개 가진 SLP 모델 초기화

in_dim  = x_train.shape[1]   # 785 (bias 포함)
out_dim = 10                 # 퍼셉트론 10개
rng = np.random.default_rng(42)
# 작은 값으로 초기화(학습 안정)
W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01

# -----------------------------
# 3) 학습 루프 (GD: 전치행렬/행렬곱만 사용)
#    - 활성화: sigmoid
#    - 손실: MSE
#    - 델타룰: ΔW = X^T * ((D - O) ⊙ O ⊙ (1 - O)) / N
# -----------------------------
EPOCHS = 30 # 학습 반복 수
LR = 0.5  # 학습 속도
for epoch in range(1, EPOCHS + 1):
    # 순전파: 입력 → 가중치 곱 → 시그모이드 통과
    Z = np.dot(x_train, W)          # 입력 × 가중치
    O = sigmoid(Z)                  # 시그모이드 통과 (출력 예측값)

    # 오차(정답 - 예측)
    delta = (y_train - O) * (O * (1.0 - O))

    # 가중치 업데이트
    grad = np.dot(x_train.T, delta) / float(x_train.shape[0])
    W = W + LR * grad # W ← W + 학습률 × 변화량


    # 학습 중 정확도 확인

    # 학습셋
    train_logits = np.dot(x_train, W)
    train_pred = np.argmax(train_logits, axis=1)
    train_acc = accuracy(y_train_int, train_pred)
    train_err = 1.0 - train_acc

    # 테스트셋
    test_logits = np.dot(x_test, W)
    test_pred = np.argmax(test_logits, axis=1)
    test_acc = accuracy(y_test_int, test_pred)
    test_err = 1.0 - test_acc

    print(f"[학습 반복 {epoch:02d}]  학습 오류율={train_err:.3f}  /  테스트 오류율={test_err:.3f}")

# -----------------------------
# 4) 최종 결과 예시 출력
# -----------------------------
print("\n샘플 10개 예측(정답 → 예측):")
for i in range(10):
    logit = np.dot(x_test[i:i+1], W)      # (1,10)
    pred  = int(np.argmax(logit, axis=1)[0])
    print(f"{int(y_test_int[i])} → {pred}")
