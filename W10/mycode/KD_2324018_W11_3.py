# -*- coding: utf-8 -*-
"""
난이도 3 실습과제 쉬운 버전
0~9 숫자를 맞추는 SLP(퍼셉트론 10개) 만들기
"""

import numpy as np
import mnist_loader as mnist  # MNIST 데이터를 불러오는 파일



# 유틸 함수

def one_hot(data, classNum):
    # 0으로 채운 (샘플 수 x 클래스 수) 크기의 ndarray 준비
    oneHotData = np.zeros((len(data), classNum)).astype('float32')

    # 각 샘플별로 정답에 해당하는 인덱스 데이터만 1로 바꿈
    for i in range(len(data)):
        oneHotData[i, data[i]] = 1.0

    return oneHotData



def sigmoid(x):
    """시그모이드: 아무 숫자든 0~1 사이 값으로 바꿔줌"""
    return 1 / (1 + np.exp(-x))


def accuracy(y_true, y_all):
    """정확도 계산: 맞춘 문제 수 / 전체 문제 수"""

    correct_count = 0   # 맞춘 문제 수 세는 변수

    # 모든 문제를 하나씩 비교
    for i in range(len(y_true)):
        if y_true[i] == y_all[i]:   # 정답과 예측이 같으면
            correct_count += 1       # 맞춘 문제 수 증가

    total = len(y_true)              # 전체 문제 수
    acc = correct_count / total      # 맞춘 비율 = 정확도

    return acc


# MNIST 데이터 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 학습 데이터는 1000개 사용
x_train = x_train[:1000] #문제 1000개
y_train = y_train[:1000] #정답 1000개

# 테스트 데이터는 MNIST 원본 10000개 모두 사용
# x_test, y_test는 그대로 사용



# 전처리

# (28x28)를 (784) 길이로 펴기
# 각 픽셀 값(0~255)을 0~1 사이로 바꿔줌 (밝기 비율)
x_train = x_train.reshape(len(x_train), 784) / 255.0  # 0~1 사이 값
x_test = x_test.reshape(len(x_test), 784) / 255.0

# bias(기본 값)을 넣기 위해 맨 앞에 1을 붙임
x_train = np.hstack([np.ones((len(x_train), 1)), x_train])  # 입력하나의 길이 784 -> 785
x_test = np.hstack([np.ones((len(x_test), 1)), x_test])

# 정수 라벨 보관 (나중에 정확도 계산용)
y_train_int = y_train.copy() # 정답 숫자 1000개
y_test_int = y_test.copy() # 정답 숫자 10000개

# one-hot 라벨 만들기
y_train = one_hot(y_train_int, 10)
y_test = one_hot(y_test_int, 10)




# SLP 퍼셉트론 10개 만들기

in_dim = x_train.shape[1]  # 785
out_dim = 10  # 숫자 0~9 → 10개 뉴런

# 가중치를 작은 랜덤값으로 시작하기
W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01



# 학습(GD)

EPOCHS = 100  # 반복 횟수
LR = 0.5  # 학습 속도(얼마나 빨리 배우는지)

for epoch in range(EPOCHS):

    # 입력(x)과 가중치(W)를 곱해 퍼셉트론의 '점수' Z 계산
    Z = np.dot(x_train, W)

    # 시그모이드로 점수를 0~1 사이 값(예측값 O)으로 변환
    O = sigmoid(Z)

    # 시그모이드 변화량 계산 (얼마나 조심해서 고칠지)
    sigmoid_deriv = O * (1 - O)

    # (정답 - 예측)의 차이 = 오차(error)
    error = y_train - O

    # 오차 × 변화량 = 가중치를 얼마나 고칠지(delta)
    delta = error * sigmoid_deriv

    # 입력(x)을 이용해 퍼셉트론 전체의 가중치 변화량(grad)을 계산
    grad = np.dot(x_train.T, delta) / len(x_train)

    # 가중치를 조금씩 수정하여 더 정답에 가깝게 만듦
    W = W + LR * grad

    # 학습 오류율 출력
    train_pred = np.argmax(np.dot(x_train, W), axis=1) # 학습셋 예측값(가장 가능성이 높은 숫자)
    train_acc = accuracy(y_train_int, train_pred) # 학습셋 정확도
    train_err = 1 - train_acc # 학습셋 오류율

    print(f"{epoch:02d}번 학습: 학습 오류율 = {train_err:.3f}")

# 학습이 끝난 뒤 테스트 오류율 출력

test_pred = np.argmax(np.dot(x_test, W), axis=1) # 테스트셋 예측값(가장 가능성이 높은 숫자)
test_acc = accuracy(y_test_int, test_pred) # 테스트셋 정확도
test_err = 1 - test_acc # 테스트셋 오류율

print("\n[최종] 테스트 오류율 =", test_err)
