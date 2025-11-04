# -*- coding: utf-8 -*-
import tensorflow as tf

# (선택) GPU 인식 여부 확인용 출력
print("GPUs:", tf.config.list_physical_devices('GPU'))

# 텐서플로 라이브러리 안의 MNIST 데이터
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# one-hot 레이블로 변환
y_train = tf.one_hot(y_train, 10).numpy()
y_test  = tf.one_hot(y_test, 10).numpy()

# 0~1 스케일링
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# SLP(퍼셉트론 10개) 모델
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

# SGD 옵티마이저 + MSE, 정확도
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
loss = 'mean_squared_error'
metrics = ['categorical_accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# (선택) 모델 구조 이미지 저장은 환경에 따라 에러날 수 있으니 필요시 주석
# tf.keras.utils.plot_model(model, 'model.png')

# 학습
model.fit(x=x_train, y=y_train, batch_size=1000, epochs=10, verbose=2)

# 평가
loss_val, acc = model.evaluate(x_test, y_test, verbose=2)
errorRate = 1.0 - acc
print('테스트 오류율: %.2f%%' % (errorRate * 100.0))
