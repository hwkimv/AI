# MNIST 데이터 세트를 읽어들여서 이미지로 출력하는 코드

# 참고: Read MNIST Dataset
# https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook

import matplotlib.pyplot as plt

import mnist_loader as mnist


# MNIST 데이터 세트 읽기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 출력할 이미지 및 레이블 지정
# 임시로 0번째 학습 데이터 샘플로 지정함
image = x_train[0]
label = y_train[0]

# 이미지 픽셀값을 콘솔에 출력
for row in range(image.shape[0]):
    # 이번 행에 출력할 문자열 구성
    rowStr = ''
    for col in range(image.shape[1]):
        rowStr += '\t%d ' % (image[row, col])
    print(rowStr)

# matplotlib으로 이미지 및 레이블 출력
plt.imshow(image, cmap=plt.cm.gray)

title_text = 'Label = %d' % label
plt.title(title_text, fontsize = 15)

plt.show()