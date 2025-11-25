# MNIST 데이터 세트를 읽어들이는 코드
import os
# 참고: Read MNIST Dataset
# https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook

import struct
import numpy as np # linear algebra
from array import array


# MNIST 데이터 세트 파일이 있는 상대 경로 (현재 스크립트 파일이 있는 경로 기준)
# 현재 파일(mnist_loader.py)의 위치
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MNIST 데이터가 있는 실제 경로: (W12/mnist)
_DATASET_BASE_PATH = os.path.join(_CURRENT_DIR, "..", "..", "mnist") + os.sep

# 각 데이터 파일의 경로
_TRAIN_IMAGE_FILE = _DATASET_BASE_PATH + 'train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _DATASET_BASE_PATH + 'train-labels.idx1-ubyte'
_TEST_IMAGE_FILE = _DATASET_BASE_PATH + 't10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _DATASET_BASE_PATH + 't10k-labels.idx1-ubyte'



# MNIST 데이터 세트를 읽어들이는 클래스
class MNISTLoader:
    # 이미지 / 레이블 파일 읽기
    def ReadImagesAndLabelFiles(self, images_filepath, labels_filepath):
        # 이미지 파일 읽기
        with open(images_filepath, 'rb') as file:
            # 바이너리 파일 형식 분석
            # size: 샘플 수 / rows x cols: 이미지 크기
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())    
        
        # 이미지 픽셀 정보를 저장할 ndarray 3차원 행렬 초기화 (0으로 채움)
        images = np.zeros((size, rows, cols)).astype('float32')
        
        # 각 이미지 샘플의 픽셀 정보를 ndarray 형식으로 변환
        for i in range(size):
            images[i, :, :] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
        
        # 레이블 파일 읽기
        with open(labels_filepath, 'rb') as file:
            # 바이너리 파일 형식 분석
            # size: 샘플 수
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            label_data = array("B", file.read())        
        
        # 레이블 정보를 ndarray 형식으로 변환
        labels = np.array(label_data).astype('int32')
                
        return images, labels
        
        
    # 학습 / 테스트 데이터 세트 모두 읽기
    def LoadData(self):
        # 학습 데이터 세트 읽기
        x_train, y_train = self.ReadImagesAndLabelFiles(_TRAIN_IMAGE_FILE, _TRAIN_LABEL_FILE)
        
        # 테스트 데이터 세트 읽기
        x_test, y_test = self.ReadImagesAndLabelFiles(_TEST_IMAGE_FILE, _TEST_LABEL_FILE)
        
        return (x_train, y_train),(x_test, y_test)    
        
        

# 데이터 로드 함수
def load_data():
    mnist_dataloader = MNISTLoader()
    return mnist_dataloader.LoadData()
