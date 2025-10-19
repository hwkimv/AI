# -*- coding: utf-8 -*-
import random


# 데이터 불러오기
def load_file(filename, label):
    """파일에서 데이터를 읽어서 리스트로 작성."""
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 줄 끝의 공백 제거
            if line == "":
                continue         # 빈 줄은 건너뜀
            parts = line.split() # 띄어쓰기 기준으로 나누기
            x = float(parts[0])  # 몸길이
            y = float(parts[1])  # 꼬리길이
            data.append([x, y, label])  # (몸길이, 꼬리길이, 라벨) 저장
    return data


def load_train_dataset():
    """학습용 데이터 읽어오기"""
    data = []
    data += load_file("salmon_train.txt", +1)   # 연어는 +1
    data += load_file("seabass_train.txt", -1)  # 농어는 -1
    return data


def load_test_dataset():
    """테스트용 데이터 읽어오기."""
    data = []
    data += load_file("salmon_test.txt", +1)
    data += load_file("seabass_test.txt", -1)
    return data



# 퍼셉트론 (선형 분류기)
def g(params, x, y):
    """퍼셉트론의 직선 식 계산"""
    a, b, c = params  # a,b: 가중치 / c: 바이오스
    return a * x + b * y + c


def predict(params, x, y):
    """퍼셉트론이 연어(+1)인지 농어(-1)인지 예측"""
    if g(params, x, y) >= 0:
        return +1   # g(x, y)가 0보다 크면 연어라고 판단
    else:
        return -1   # 작으면 농어라고 판단


def error(params, dataset):
    """오류율 계산"""
    wrong = 0
    for x, y, t in dataset:  # (x, y, t): 몸길이, 꼬리길이, 정답라벨
        pred = predict(params, x, y)
        if pred != t:  # 예측값과 실제 정답이 다르면 오류
            wrong += 1
    return wrong / len(dataset)  # 틀린 개수 / 전체 개수


def fitness(cost_value):
    """적합도"""
    return 1.0 / (1.0 + cost_value)



# 유전 알고리즘(GA)
def random_param():
    """-1 ~ 1 사이의 무작위 수 하나를 만들기"""
    return random.uniform(-1, 1)


def make_individual():
    """개체 하나 만들기: [a, b, c]"""
    return [random_param(), random_param(), random_param()]


def crossover(p1, p2):
    """부모 두 개체를 섞어서 자식 개체 만들기."""
    child = []
    for i in range(3):
        if random.random() < 0.5:   # 50% 확률로 부모1의 값 선택
            child.append(p1[i])
        else:                       # 나머지는 부모2의 값 선택
            child.append(p2[i])
    return child


def mutate(ind):
    """개체를 10% 확률로 살짝 바꾼다(돌연변이)."""
    for i in range(3):
        if random.random() < 0.1: # 10% 확률로 돌연변이 발생
            ind[i] = random_param() # 해당 유전자를 무작위 값으로 변경


def select_parent(pop, fits):
    """적합도에 따라 부모를 고르기. (룰렛 선택 방식)"""
    total = 0
    for f in fits:
        total += f  # 전체 적합도 합

    r = random.uniform(0, total)  # 0 ~ total 사이의 무작위 값
    acc = 0
    for i in range(len(pop)):
        acc += fits[i]  # 누적 합이 무작위값보다 커지면
        if acc >= r:
            return pop[i]  # 해당 개체를 선택 (확률적으로 좋은 개체가 잘 뽑힘)



# 메인 학습
def main():
    random.seed(1)  #시드 고정

    # 학습 데이터, 테스트 데이터 불러오기
    train_data = load_train_dataset()
    test_data = load_test_dataset()

    POP = 40          # 한 세대에 있는 개체 수
    GENERATIONS = 100 # 총 세대(반복) 수

    # 처음 세대(무작위 개체들)
    pop = []
    for i in range(POP):
        pop.append(make_individual()) # 개체 추가

    # 세대별 학습 반복 시작
    for gen in range(1, GENERATIONS + 1):
        costs = []  # 각 개체의 오류율
        fits = []   # 각 개체의 적합도

        # 모든 개체를 평가 (학습 데이터 기준)
        for ind in pop:
            c = error(ind, train_data)  # 학습 데이터에서 틀린 비율 계산
            f = fitness(c)              # 오류율이 낮을수록 fitness는 커짐
            costs.append(c)
            fits.append(f)

        # 이번 세대에서 가장 좋은 개체 찾기
        best_idx = 0
        for i in range(1, POP):
            if fits[i] > fits[best_idx]: # 적합도가 더 높으면
                best_idx = i

        # 최고 개체 정보 (오류율, 파라미터)
        best_params = pop[best_idx]
        train_err = error(best_params, train_data)  # 가장 좋은 개체의 학습 오류율
        test_err = error(best_params, test_data)    # 가장 좋은 개체의 테스트 오류율

        # 출력 형식
        print("[세대", gen, "]",
              "학습 오류율 =", round(train_err * 100, 1), "% |",
              "테스트 오류율 =", round(test_err * 100, 1), "%")

        # 다음 세대 만들기 (엘리트 보존)
        new_pop = []
        new_pop.append(pop[best_idx][:])  # 가장 잘한 개체는 그대로 복사

        # 나머지 개체들 생성
        while len(new_pop) < POP:
            p1 = select_parent(pop, fits) # 부모1 선택
            p2 = select_parent(pop, fits) # 부모2 선택
            child = crossover(p1, p2) # 교차로 자식 생성
            mutate(child) # 돌연변이 적용
            new_pop.append(child) # 새 개체 추가

        pop = new_pop  # 세대 교체

# 프로그램 실행 시작
if __name__ == "__main__":
    main()
