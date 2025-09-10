# -*- coding: utf-8 -*-
"""
Simulated Annealing으로 선형 분류기 학습 (난이도 3 과제 스펙)

- 데이터: salmon_train.txt, seabass_train.txt  (각 줄: 몸길이 꼬리길이)
- 모델: 선형 결정경계  f(x, y) = a*x + b*y + c
- 초기 파라미터: [2.0, -1.0, -180.0]
- 초기 온도: T = 100
- 냉각: T *= 0.99,  T < 0.001 이면 종료
- 이웃 이동: a += U(-0.01, +0.01), b += U(-0.01, +0.01), c += U(-10.0, +10.0)
- 목적함수(에너지): 분류 오류율 (낮을수록 좋음)
- SA 수용 규칙: ΔE <= 0 이면 항상 수용, 그렇지 않으면 확률 exp(-ΔE / T)로 수용
- 매 반복마다 현재 오류율을 출력
"""

import random
import math

SALMON_FILE = "salmon_train.txt"
SEABASS_FILE = "seabass_train.txt"

# ----------------------------------
# 1) 데이터 로드
# ----------------------------------
def load_data():
    def read_xy(path, label):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                x_str, y_str = line.split()
                x, y = float(x_str), float(y_str)
                items.append((x, y, label))
        return items

    # salmon=1, seabass=0 로 라벨링
    data = []
    data += read_xy(SALMON_FILE, 1)
    data += read_xy(SEABASS_FILE, 0)
    random.shuffle(data)
    return data

# ----------------------------------
# 2) 선형 분류기와 오류율
# ----------------------------------
def predict(a, b, c, x, y):
    """f(x,y) = a*x + b*y + c >= 0 이면 연어(1), 아니면 농어(0)"""
    return 1 if (a*x + b*y + c) >= 0.0 else 0

def error_rate(params, dataset):
    """분류 오류율(= 1 - 정확도)"""
    a, b, c = params
    wrong = 0
    for x, y, label in dataset:
        if predict(a, b, c, x, y) != label:
            wrong += 1
    return wrong / len(dataset)

# ----------------------------------
# 3) 이웃(Neighbor) 생성
# ----------------------------------
def neighbor(params):
    """과제 조건의 이동 구간으로 랜덤 이웃 생성"""
    a, b, c = params
    a += random.uniform(-0.01, +0.01)
    b += random.uniform(-0.01, +0.01)
    c += random.uniform(-10.0, +10.0)
    return [a, b, c]

# ----------------------------------
# 4) Simulated Annealing
# ----------------------------------
def simulated_annealing(dataset,
                        init_params=[2.0, -1.0, -180.0],
                        T_init=100.0,
                        T_min=1e-3,
                        alpha=0.99,
                        iters_per_T=1):
    """
    iters_per_T: 온도 한 번에 몇 번의 이웃 탐색을 할지
    """
    current = list(init_params)
    current_E = error_rate(current, dataset)
    T = T_init
    step = 0

    print(f"[시작] 초기 온도={T:.3f}  초기 파라미터={current}  초기 오류율={current_E:.4f}")

    while T >= T_min:
        for _ in range(iters_per_T):
            step += 1
            cand = neighbor(current)
            cand_E = error_rate(cand, dataset)
            dE = cand_E - current_E

            # SA 수용 규칙
            if dE <= 0:
                accept = True
            else:
                accept_prob = math.exp(-dE / T)
                accept = (random.random() < accept_prob)

            if accept:
                current, current_E = cand, cand_E

            # 매 반복마다 오류율 출력
            print(f"[반복={step:5d}]  온도={T:7.4f}  오류율={current_E:.4f}  파라미터={current}")

        # 냉각
        T *= alpha

    return current, current_E

# ----------------------------------
# 5) 실행
# ----------------------------------
if __name__ == "__main__":
    random.seed(42)  # 재현성(원하면 주석 처리)
    data = load_data()
    best_params, best_err = simulated_annealing(data)
    a, b, c = best_params
    print("\n[최종 결과]")
    print(f"최적 파라미터: a={a:.6f}, b={b:.6f}, c={c:.6f}")
    print(f"최종 오류율: {best_err:.4f}")
    print("결정 경계 식: a*x + b*y + c = 0")
    print("예측 규칙: a*x + b*y + c >= 0 이면 '연어', 그렇지 않으면 '농어'")
