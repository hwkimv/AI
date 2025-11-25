# MNIST DB의 0~9 숫자 중 0만 검출하는 퍼셉트론(perceptron)을 학습하는 예시 코드
# 이 퍼셉트론은 숫자 0의 이미지가 입력되면 1을 출력
# 이 퍼셉트론은 숫자 1~9의 이미지가 입력되면 0을 출력
# 비용 함수: 평균 제곱 오차(mean squared error)
# 학습 알고리즘: SGD(stochastic gradient descent) 미니 배치(mini batch) 방식

import copy as cp
import numpy as np
import numpy.random as nr
import pickle as pkl

import mnist_loader as mnist

# 랜덤 시드 고정
nr.seed(12345)



# 모든 실험에서 고정해서 쓰는 상수들
_TRAIN_SAMPLE_NUM = 2000            # 모델 학습 세트로 사용할 샘플 수
_VALID_SAMPLE_NUM = 1000            # 검증(validation) 세트로 사용할 샘플 수


# [실습해볼 내용] 검증 오류율을 더 낮출 수 있는 하이퍼 파라미터 찾아보기


# 반복 실험하며 최적화가 필요한 하이퍼 파라미터들
_INIT_PARAM_RANGE = 0.001             # 파라미터 랜덤 초기화 범위 (-_INIT_PARAM_RANGE ~ _INIT_PARAM_RANGE)
_INIT_LEARNING_RATE = 0.1          # 초기 학습률
_LEARNING_RATE_DECAY_FACTOR = 0.5   # 학습률에 대한 감쇠율
_BATCH_SIZE = 1                  # 한 배치 내 샘플 수


# 학습 과정에 일부 영향을 줄 수도 있는 그 밖의 설정값들
_MAX_EPOCH = 1000                   # 최대 에포크 횟수 (이 전에라도 검증 오류율이 증가하면 조기 종료될 수 있음)
_VALID_INTERVAL_EPOCH = 10          # 일정 횟수의 에포크를 학습할 때마다 검증 수행
_EARLY_STOPPING_PATIENT = 1         # 검증 오류율이 감소하지 않더라도 조금 더 학습해볼 횟수



# 0 검출기를 학습하기 위해 0~9 사이의 원본 숫자 레이블을 다음과 같이 변환
# 원본이 0일 때 -> 1
# 원본이 1, 2, 3, 4, 5, 6, 7, 8, 9일 때 -> 0
# oldLabels: 원본 숫자 레이블 데이터(ndarray 자료형)
def convertLabel(oldLabels):
    newLabels = []
    for i in range(len(oldLabels)):
        # 원본이 0일 때 -> 1
        if oldLabels[i] == 0:
            newLabels.append(1.0)
        # 원본이 1, 2, 3, 4, 5, 6, 7, 8, 9일 때 -> 0
        else:
            newLabels.append(0.0)

    return np.array(newLabels)



# 로지스틱 시그모이드 함수
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))



# 모델 검증 / 성능 평가
def evaluate(model, x_array, y_array):
    sampleNum = x_array.shape[0]    # 샘플 수
    squaredErrorSum = 0.0           # 제곱 오차 누적치
    correctCnt = 0                  # 정답을 맞춘 갯수 누적치

    # 각 샘플별로 인식이 잘 되었는지 오차 계산
    for i in range(sampleNum):
        # i번째 샘플의 특징 벡터 x
        x = x_array[i]

        # i번째 샘플의 정답 레이블 y: 0 혹은 1
        y = y_array[i]

        # i번째 샘플에 대한 모델의 출력 o: 0~1 사이의 실수값
        o = model.CalcActivation(x)

        # 제곱 오차 합산
        squaredErrorSum += (o - y) * (o - y)

        # 모델의 출력을 0 혹은 1로 최종 결정
        # 0.5보다 크면 1, 작으면 0 (일종의 계단 함수)
        if o >= 0.5:
            finalResult = 1
        else:
            finalResult = 0

        # 정답을 맞췄으면 정답 카운트 +1
        if finalResult == y:
            correctCnt += 1

    # 평균 제곱 오차(MSE, mean squared error)
    mse = squaredErrorSum / float(sampleNum)

    # 오류율(error rate)
    errorRate = 1.0 - (float(correctCnt) / float(sampleNum))

    return mse, errorRate



# 모델: Perceptron
class Model:
    inputDim = None         # 입력 특징의 차원
    w = None                # 파라미터 (0번째는 바이어스)
    delta = None            # 그래디언트를 누적한 파라미터 변화량
    learningRate = None     # 학습률


    # 모델 클래스 생성자
    # inputDim: 입력될 특징의 차원
    def __init__(self, inputDim):
        self.inputDim = inputDim

        # 파라미터 랜덤 초기화 (0번째는 바이어스)
        self.w = nr.rand(self.inputDim + 1).astype('float32') * _INIT_PARAM_RANGE

        # 파라미터 변화량은 0으로 초기화
        self.delta = np.zeros(self.inputDim + 1).astype('float32')

        # 학습률은 초기 학습률로 시작
        self.learningRate = _INIT_LEARNING_RATE


    # 모델의 출력(활성값) 계산
    def CalcActivation(self, x):
        # 모델 파라미터 w와 입력 특징 벡터 x를 내적
        # z = dot(w, x)
        z = np.sum(self.w * x)

        # 활성 함수를 거친 활성값 계산
        # o = activation(z)
        o = logistic(z)

        return o


    # 한 샘플에 대한 delta 누적
    # x: 샘플의 특징
    # y: 샘플의 레이블
    # batchSize: 배치 크기(한 배치 당 샘플 수)
    def TrainOneSample(self, x, y):
        # 모델의 출력 o 계산
        o = self.CalcActivation(x)

        # 제곱 오차 비용함수를 미분한 gradient 계산
        gradient = - (y - o) * o * (1.0 - o) * x

        # 배치 크기가 너무 클 경우 누적 그래디언트의 값이 너무 커져서 학습이 제대로 수행되지 않을 수 있음
        # 배치 크기로 나누어 평균 그래디언트로 계산하기
        gradient /= float(_BATCH_SIZE)

        # 파라미터에 더할 값(변화량) 계산
        # - 학습률 * gradient
        self.delta += - self.learningRate * gradient


    # 누적해 둔 delta를 파라미터에 더하기(한 batch가 끝난 뒤 호출)
    def Update(self):
        self.w += self.delta

        # 파라미터 변화량은 0으로 초기화
        self.delta = np.zeros(self.inputDim + 1).astype('float32')


    # 학습률 감쇠 (한 epoch가 끝난 뒤 호출)
    def DecayLearningRate(self):
        self.learningRate *= _LEARNING_RATE_DECAY_FACTOR



# MNIST 데이터 세트 읽기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 특징이 0~1 사이의 값이 되도록 scaling
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2차원 이미지 픽셀들을 1차원으로 길게 연결(flatten)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# 특징 벡터의 차원(바이어스에 곱할 1을 패딩하기 전)
inputDim = x_train.shape[1]
print('inputDim:', inputDim)

# 바이어스에 곱해질 1을 입력 특징 맨 앞에 패딩(padding, 끼워넣음)
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

# 0 검출기를 학습하기 위해 0~9 사이의 원본 숫자 레이블을 다음과 같이 변환
# 원본이 0일 때 -> 1
# 원본이 1, 2, 3, 4, 5, 6, 7, 8, 9일 때 -> 0
y_train = convertLabel(y_train)
y_test = convertLabel(y_test)



# 학습 데이터 세트: MNIST train set 중 앞에서 2000개 샘플
# 검증 데이터 세트: MNIST train set 중 뒤에서 1000개 샘플
# 평가 데이터 세트: MNIST test set 전체(10000개 샘플)
x_valid = x_train[:_VALID_SAMPLE_NUM]
y_valid = y_train[:_VALID_SAMPLE_NUM]

x_train = x_train[-_TRAIN_SAMPLE_NUM:]
y_train = y_train[-_TRAIN_SAMPLE_NUM:]

# 학습 데이터세트 데이터 유형 및 형태
print('x_train:', type(x_train), x_train.shape)
print('y_train:', type(y_train), y_train.shape)

# 검증 데이터세트 데이터 유형 및 형태
print('x_valid:', type(x_valid), x_valid.shape)
print('y_valid:', type(y_valid), y_valid.shape)

# 평가 데이터세트 데이터 유형 및 형태
print('x_test:', type(x_test), x_test.shape)
print('y_test:', type(y_test), y_test.shape)



# 모델 준비
model = Model(inputDim)

# 학습 시작 전 검증 데이터 세트에 대한 비용(평균 제곱 오차), 오류율 측정
validMSE, validErrorRate = evaluate(model, x_valid, y_valid)
output = 'before training\tvalidMSE=%.6f\tvalidErrorRate=%.2f%%' % (validMSE, validErrorRate * 100)
print(output, flush=True)

# 최적 검증 오류율 초기화
bestValidMSE = validMSE
bestValidErrorRate = validErrorRate

# 한 epoch당 몇 회의 batch 학습이 이루어지는지 계산
# 횟수가 딱 나누어 떨어지지 않을 경우 짜투리 샘플들은 버림
maxBatchCnt = int(float(_TRAIN_SAMPLE_NUM) / float(_BATCH_SIZE))


# epoch 반복
patientCnt = 0          # early stopping되기까지 좀 더 지켜볼 횟수
for epoch in range(1, _MAX_EPOCH + 1):
    # 학습 데이터 세트에 대한 배치 학습 반복
    for batchCnt in range(maxBatchCnt):
        batchStartSampleIndex = _BATCH_SIZE * batchCnt

        # 한 배치 내의 샘플을 하나씩 처리
        for sampleCnt in range(_BATCH_SIZE):
            currentSampleIndex = batchStartSampleIndex + sampleCnt
            model.TrainOneSample(x_train[currentSampleIndex], y_train[currentSampleIndex])

        # 한 배치 내의 샘플을 모두 본 후 파라미터 업데이트
        model.Update()

    # 학습률 감쇠
    model.DecayLearningRate()

    # 검증을 수행할 에포크인가?
    if epoch % _VALID_INTERVAL_EPOCH == 0:
        # 검증 데이터 세트에 대한 비용(평균 제곱 오차), 오류율 측정
        validMSE, validErrorRate = evaluate(model, x_valid, y_valid)
        output = 'epoch=%04d\tvalidMSE=%.6f\tvalidErrorRate=%.2f%%' % (epoch, validMSE, validErrorRate * 100)
        print(output, flush=True)


        # [실습해볼 내용] 화면에 출력되는 내용들을 로그 파일에도 기록하기


        # 검증 데이터 세트에 대한 비용이 낮아졌다면
        if validMSE < bestValidMSE:
            # 최적 검증 오류율 갱신
            bestValidMSE = validMSE
            bestValidErrorRate = validErrorRate

            # [실습해볼 내용] 최적 모델을 파일에 저장하기
            flie = open('model.pkl', 'wb')
            pkl.dump(model, flie)
            flie.close()

        # 검증 오류율이 낮아지지 않았다면
        else:
            patientCnt += 1

            # 학습을 조기 중단(Early Stopping)할 것인가?
            if patientCnt > _EARLY_STOPPING_PATIENT:
                break


# 학습 종료 후 검증 데이터 세트에 대한 최적 비용(평균 제곱 오차) 및 오류율 출력
output = 'after training\tbestValidMSE=%.6f\tbestValidErrorRate=%.2f%%' % (bestValidMSE, bestValidErrorRate * 100)
print(output, flush=True)

# ---- 여기부터 로그 파일에 기록 ----
VERSION = "v3"  # 버전명만 실행할 때마다 바꿔주면 됨

with open("train_result.txt", "a", encoding="utf-8") as f:
    f.write(
        f"version={VERSION}\t"
        f"INIT_RANGE={_INIT_PARAM_RANGE}\t"
        f"INIT_LR={_INIT_LEARNING_RATE}\t"
        f"DECAY={_LEARNING_RATE_DECAY_FACTOR}\t"
        f"BATCH={_BATCH_SIZE}\t"
        f"bestValidMSE={bestValidMSE:.6f}\t"
        f"bestValidErrorRate={bestValidErrorRate:.6f}\t"
        f"MODEL=model.pkl\n"
    )

# [실습해볼 내용] 모델 학습 스크립트와 성능 평가 실험 스크립트를 따로 분리하기