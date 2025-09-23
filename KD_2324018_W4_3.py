# -*- coding: utf-8 -*-
"""
간단한 GA(유전 알고리즘)으로 선형 분류기 학습하기
 - 연어(salmon) vs 농어(seabass) 구분
 - 몸길이, 꼬리길이 데이터 사용
"""

import random

'''데이터 로드'''
# 파일에서 (x, y, label) 읽어오기
def load_file(filename, label):
    data = []
    f = open(filename, "r", encoding="utf-8")
    for line in f:
        line = line.strip()
        if line == "":
            continue
        parts = line.split()
        x = float(parts[0])
        y = float(parts[1])
        data.append([x, y, label])
    f.close()
    return data

def load_dataset():
    data = []
    data += load_file("salmon_train.txt", +1)   # 연어
    data += load_file("seabass_train.txt", -1)  # 농어
    return data

'''선형 분류기'''
# 결정함수
def g(params, x, y):
    a, b, c = params
    return a * x + b * y + c

# 예측
def predict(params, x, y):
    if g(params, x, y) >= 0:
        return +1   # 연어
    else:
        return -1   # 농어

# 손실함수
def cost(params, dataset):
    total = len(dataset)
    s = 0
    for x, y, t in dataset:
        pred = predict(params, x, y)   # 예측
        gx = g(params, x, y)           # 결정함수 값
        s += abs(pred - t) * abs(gx)   # 절댓값: 오차유무(예측-정답) × 오차 크기
    return s / total

# 적합도
def fitness(cost):
    return 1.0 / (1.0 + cost)

#오류율
def error(params, dataset):
    wrong = 0
    for x, y, t in dataset:
        pred = predict(params, x, y)
        if pred != t:
            wrong += 1
    return wrong / len(dataset)


'''GA 유전 알고리즘'''
# 랜덤 파라미터 생성
def random_param():
    return random.uniform(-1, 1) #-1~1 사이의 실수

# 개체 생성
def make_individual():
    return [random_param(), random_param(), random_param()]

# 교차: 두 부모의 유전자를 절반씩 섞음
def crossover(p1, p2):
    child = [] # 자식 개체
    for i in range(3):
        if random.random() < 0.5: # 50% 확률로 부모1의 유전자 선택
            child.append(p1[i])
        else:                     # 50% 확률로 부모2의 유전자 선택
            child.append(p2[i])
    return child

# 돌연변이: 각 유전자에 대해 10% 확률로 랜덤값으로 변경
def mutate(ind): # ind: 개체
    for i in range(3):
        if random.random() < 0.1:   # 10% 확률
            ind[i] = random_param()

# 선택: 부모 개체를 선택(룰렛)
def select_parent(pop, fits): # pop: 개체 목록, fits: 적합도 목록
    # 적합도 총합 계산
    total = 0
    for f in fits:
        total += f

    # 룰렛 선택
    r = random.uniform(0, total) # 0~total 사이의 랜덤값
    acc = 0 # 룰렛 누적값
    for i in range(len(pop)): # 룰렛 선택
        acc += fits[i] # 룰렛 적합도 갱신
        if acc >= r: # 누적값이 랜덤값 이상이면 선택
            return pop[i] # 선택된 개체 반환


'''메인 함수'''
def main():
    random.seed(1)

    dataset = load_dataset() # 데이터 읽기
    print("전체 데이터 개수:", len(dataset))
    print("데이터 샘플 :")
    for i in range(min(5, len(dataset))):
        x, y, t = dataset[i]
        label = "연어" if t == +1 else "농어"
        print(" ", i+1, ": 몸길이=", x, "꼬리길이=", y, "레이블=", label)

    # 초기 세대
    POP = 25 # 개체 수
    pop = [] # 개체 목록
    for i in range(POP): # 초기 개체 생성
        pop.append(make_individual())
    print("\n[첫세대] 초기 개체들:")

    costs = [] # 비용(손실)
    fits = [] # 적합도
    for i in range(POP):
        c = error(pop[i], dataset) # 오류율 계산
        f = fitness(c)  # 적합도 계산
        costs.append(c) # 비용 목록에 추가
        fits.append(f) # 적합도 목록에 추가
        print("  개체", i, ":", pop[i], "분류 오류율=", round(c,4), "적합도=", round(f,6))

    # 첫세대 의 최고 개체 찾기
    best_idx = 0
    for i in range(1, POP):
        if fits[i] > fits[best_idx]: #적합도가 더 높으면
            best_idx = i

    print("\n[첫세대 요약] 최고개체=", best_idx,
          "분류 오류율=", round(costs[best_idx],4),
          "적합도=", round(fits[best_idx],6))

    # 세대 반복
    GENERATIONS = 100
    for gen in range(1, GENERATIONS + 1):

        costs = []
        fits = []
        
        for ind in pop: # 각 개체에 대해
            c = error(ind, dataset)
            f = fitness(c)
            costs.append(c)
            fits.append(f)

        # 최고 개체 찾기
        best_idx = 0
        for i in range(1, POP):
            if fits[i] > fits[best_idx]:
                best_idx = i

        print("[세대", gen, "] 최고개체=", best_idx,
              "분류 오류율=", round(costs[best_idx],4),
              "적합도=", round(fits[best_idx],6))

        # 마지막 세대면 종료
        if gen == GENERATIONS:
            break

        # 다음 세대 만들기 (엘리트 보존)
        new_pop = []
        new_pop.append(pop[best_idx][:])   # 최고 개체 복사

        while len(new_pop) < POP: # 새 개체가 POP개 될 때까지
            p1 = select_parent(pop, fits) # 부모 선택
            p2 = select_parent(pop, fits)  # 부모 선택
            child = crossover(p1, p2) # 교차
            mutate(child) # 돌연변이
            new_pop.append(child) # 새 개체 추가

        pop = new_pop # 세대 교체

    # 최종 결과
    final_best = pop[best_idx]
    print("\n최종 파라미터:", final_best)
    print("최종 오류율:", round(costs[best_idx], 6))

if __name__ == "__main__":
    main()
