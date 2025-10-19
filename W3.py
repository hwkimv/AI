# -*- coding: utf-8 -*-
"""
Simulated Annealing 으로 선형 분류기 학습

[문제 설정]
- 두 데이터 파일에서 연어(salmon), 농어(seabass)의 (몸길이, 꼬리길이) 샘플을 읽어온다.
- 선형 결정경계 f(x, y) = a*x + b*y + c 를 학습하여 둘을 분리한다.

[최적화 방식: Simulated Annealing (모의 담금질)]
- 에너지(목적함수) = 분류 오류율 (낮을수록 좋은 해)
- 이웃 이동: (a, b, c)에 작은 랜덤 노이즈를 더해 새 후보 해(θ_new)를 만든다.
- 수용 규칙:
    ΔE = E_new - E_current
    - ΔE <= 0 이면(개선) 항상 수용
    - ΔE  > 0 이면(악화) 확률 exp(-ΔE / T) 로 수용  ← 지역최적해 탈출 장치
- 온도 T 는 단계마다 α배(0<α<1)로 감소(냉각)하며, T < T_min 이면 종료.

[출력]
- 매 반복마다 현재 오류율과 파라미터를 출력(과제 스펙).
- 마지막 상태와 최고 성능(best) 상태를 모두 보고한다.
"""

import random
import math

# ----- 파일 경로 -----
# 학습 데이터가 저장된 텍스트 파일 이름 (작업 디렉토리에 두는 것을 가정)
SALMON_FILE = "salmon_train.txt"
SEABASS_FILE = "seabass_train.txt"

# ----- 과제 기본 하이퍼파라미터 -----
# 선형 결정경계의 초기 파라미터 (a, b, c)
INIT_PARAMS = [2.0, -1.0, -180.0]

# SA(모의 담금질)의 초기 온도, 종료 온도 기준, 냉각 비율
T_INIT      = 100.0   # 시작 온도 (높을수록 초반에 나빠진 해도 더 잘 수용 → 탐색 폭↑)
T_MIN       = 1e-3    # 이 값보다 작아지면 종료 (부동소수점 오차로 인한 무한루프 방지)
ALPHA       = 0.99    # 한 온도 단계가 끝날 때마다 T ← ALPHA * T
ITERS_PER_T = 1       # 온도 단계당 이웃 탐색 횟수 (과제 스펙: 1)

# ----- 이웃 이동 범위(스펙 값) -----
# a, b는 기울기(특징 스케일이 커질수록 민감), c는 절편(평행 이동)이라 폭을 더 크게 둔다.
DA = 0.01
DB = 0.01
DC = 10.0


# 1) 데이터 로드 -------------------------------------------------------------
def load_data():
    """
    두 파일에서 (x, y, label)을 읽어 하나의 리스트로 합쳐 반환한다.
    - label: salmon → 1, seabass → 0 (이진분류용 라벨 인코딩)
    - 각 줄 형식: "<몸길이> <꼬리길이>"
    """
    def read_xy(path, label):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:           # 공백/빈 줄은 건너뛰기
                    continue
                # 파일은 "정수 정수" 혹은 "실수 실수"라고 가정
                x, y = map(float, line.split())
                rows.append((x, y, label))
        return rows

    # 두 클래스 데이터를 합치고, 샘플 순서를 섞어(셔플) 편향을 줄인다.
    data = read_xy(SALMON_FILE, 1) + read_xy(SEABASS_FILE, 0)
    random.shuffle(data)
    return data


# 2) 모델과 오류율 -----------------------------------------------------------
def predict(a, b, c, x, y):
    """
    선형 결정함수 f(x,y) = a*x + b*y + c
    - f >= 0 이면 salmon(1), f < 0 이면 seabass(0)로 판정한다.
    """
    return 1 if (a*x + b*y + c) >= 0.0 else 0

def error_rate(params, dataset):
    """
    분류 오류율(= 틀린 개수 / 전체 개수)을 계산한다.
    - Simulated Annealing에서 '에너지'로 사용되며, 낮을수록 좋은 해다.
    """
    a, b, c = params
    wrong = 0
    for x, y, t in dataset:
        pred = predict(a, b, c, x, y)
        if pred != t:
            wrong += 1
    return wrong / len(dataset)


# 3) 이웃 생성 ---------------------------------------------------------------
def neighbor(params):
    """
    현재 파라미터에서 작은 랜덤 이동을 적용해 새 후보 파라미터(이웃 해)를 만든다.
    - 과제 스펙의 범위에 맞춰 a, b, c 각각에 균등분포 U(-Δ, +Δ) 노이즈를 더한다.
    - 탐색 폭(DA/DB/DC)은 데이터 스케일/난이도에 따라 튜닝 가능.
    """
    a, b, c = params
    return [
        a + random.uniform(-DA, +DA),
        b + random.uniform(-DB, +DB),
        c + random.uniform(-DC, +DC),
        ]


# 4) Simulated Annealing 본체 ------------------------------------------------
def simulated_annealing(dataset):
    """
    Simulated Annealing 루프를 수행한다.

    반환:
    - last_params: 루프 종료 시점(마지막 온도)의 파라미터
    - last_err   : 마지막 시점의 오류율
    - best_params: 탐색 전체 과정에서 관측된 최저 오류율 파라미터(Report용으로 유용)
    - best_err   : 그때의 오류율
    """
    # 현재 해(θ)와 그 에너지(E)를 초기화
    current = INIT_PARAMS[:]                 # 리스트 복사 (원본 보호)
    curr_E = error_rate(current, dataset)

    # 최고 성능(best) 추적 (항상 마지막 해가 최고와 같다는 보장은 없기 때문)
    best_params = current[:]
    best_E = curr_E

    # 온도 초기화
    T = T_INIT
    step = 0  # 반복 카운터(디버깅/로그용)
    print(f"[시작] T={T:.3f}  params={current}  err={curr_E:.4f}")

    # 종료 조건: T < T_MIN (수치오차 예방을 위해 '==' 대신 임계값 사용)
    while T >= T_MIN:
        # 온도 단계 내에서 이웃 탐색을 몇 번 할지 (과제 스펙: 1)
        for _ in range(ITERS_PER_T):
            step += 1

            # 1) 이웃 해 생성
            cand = neighbor(current)

            # 2) 후보 해의 에너지(오류율) 계산
            cand_E = error_rate(cand, dataset)

            # 3) 에너지 차이 ΔE = E_new - E_current
            #    - ΔE <= 0: 개선 → 항상 수용
            #    - ΔE  > 0: 악화 → 확률 exp(-ΔE/T)로 수용 (온도가 높을수록 잘 수용)
            dE = cand_E - curr_E

            if dE <= 0:
                accept = True
            else:
                # math.exp(-dE / T)가 0~1 사이 확률이 되어 수용 여부를 결정
                # T가 낮아질수록(냉각 진행) 나빠진 해를 덜 수용 → 수렴
                p = math.exp(-dE / T)
                accept = (random.random() < p)

            # 4) 수용 시 현재 해/에너지 갱신
            if accept:
                current, curr_E = cand, cand_E

                # 5) '최고 성능' 갱신(리포트/재현을 위해 별도로 기록)
                if curr_E < best_E:
                    best_params, best_E = current[:], curr_E

            # (과제 스펙) 매 반복마다 현재 오류율과 파라미터를 출력
            print(f"[반복={step:5d}] T={T:7.4f}  err={curr_E:.4f}  params={current}")

        # 6) 냉각: 다음 라운드로 갈수록 T가 줄어들며 점차 탐색보다 수렴에 무게
        T *= ALPHA

    # 마지막 상태(종료 시점) + 탐색 전체에서의 최고 상태를 함께 반환
    return current, curr_E, best_params, best_E


# 5) 실행부 ------------------------------------------------------------------
if __name__ == "__main__":
    # 난수 시드 고정: 실험을 반복해도 같은 결과가 재현되도록(원하면 주석 처리)
    random.seed(42)

    # 데이터 로드
    data = load_data()

    # SA 실행
    last_params, last_err, best_params, best_err = simulated_annealing(data)

    # 요약 리포트
    a, b, c = best_params
    print("\n[최종 보고]")
    # 종료 시점 상태(마지막 온도에서의 결과)
    print(f"- 마지막 파라미터: {last_params}, 마지막 오류율: {last_err:.4f}")
    # 탐색 중 가장 좋았던 상태(최소 오류율)
    print(f"- 최고(최소) 오류율 파라미터: a={a:.6f}, b={b:.6f}, c={c:.6f}")
    print(f"- 최고(최소) 오류율: {best_err:.4f}")
    print("- 결정 경계: a*x + b*y + c = 0")
    print("- 예측 규칙: f(x,y) >= 0 → 'salmon', 그 외 'seabass'")
