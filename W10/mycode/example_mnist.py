import numpy as np

import mnist_loader as mnist


# 0~9 사이의 자연수 레이블을 one-hot 벡터로 변환
# data: 원본 자연수 레이블 데이터(1차원 ndarray)
# classNum: 클래스 수(ex> 0~9 숫자 인식기의 경우 클래스는 10개)
def one_hot(data, classNum):
    # 0으로 채운 (샘플 수 x 클래스 수) 크기의 ndarray 준비
    oneHotData = np.zeros((len(data), classNum)).astype('float32')
    
    # 각 샘플별로 정답에 해당하는 인덱스 데이터만 1로 바꿈
    for i in range(len(data)):
        oneHotData[i, data[i]] = 1.0
    
    return oneHotData
    

# MNIST 데이터 세트 읽기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 학습 데이터세트(특징, 60000개, 28x28 픽셀)
print('x_train:', type(x_train), x_train.shape)

# 학습 데이터세트(정답 레이블, 60000개)
print('y_train:', type(y_train), y_train.shape)

# 테스트 데이터세트(특징, 60000개, 28x28 픽셀)
print('x_test:', type(x_test), x_test.shape)

# 테스트 데이터세트(정답 레이블, 10000개)
print('y_test:', type(y_test), y_test.shape)

# 특징이 0~1 사이의 값이 되도록 scaling
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2차원 이미지 픽셀들을 1차원으로 길게 연결(flatten)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# 입력 특징 맨 앞에 bias term에 대응되는 1 패딩
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

print('학습 데이터 샘플 첫 5개의 숫자 레이블')
print(y_train[:5])

# ont-hot representation
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

print('학습 데이터 샘플 첫 5개의 one-hot 레이블')
print(y_train[:5])


