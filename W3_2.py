# -*- coding: utf-8 -*-

SALMON_FILE = "salmon_train.txt"   # 연어 데이터 (라벨 = 1)
SEABASS_FILE = "seabass_train.txt" # 농어 데이터 (라벨 = 0)

# 선형 분류기 파라미터 (슬라이드 고정값)
A, B, C = 2.0, -1.0, -180.0

def predict(a, b, c, x, y):
    """결정함수 f(x,y)=a*x+b*y+c >= 0 이면 연어, 아니면 농어"""
    return 1 if (a*x + b*y + c) >= 0 else 0

def read_dataset(path, label):
    """텍스트 파일을 읽어 (몸, 꼬리, 라벨) 튜플 목록으로 반환"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            bx, ty = s.split()
            items.append((float(bx), float(ty), label))
    return items

def main():
    # 1) 데이터 읽기
    data = []
    data += read_dataset(SALMON_FILE, 1)  # 연어
    data += read_dataset(SEABASS_FILE, 0) # 농어

    # 2) 한 마리씩 출력 + 분류 + 정답 비교
    wrong = 0
    total = 0
    for i, (body, tail, label) in enumerate(data, start=1):
        pred = predict(A, B, C, body, tail)
        pred_name = "연어" if pred == 1 else "농어"
        true_name = "연어" if label == 1 else "농어"
        is_correct = (pred == label)

        print(f"물고기{i:03d} | 몸={body:.1f}, 꼬리={tail:.1f} -> 예측: {pred_name} / 정답: {true_name} "
              f"{'✔' if is_correct else '✘'}")

        total += 1
        wrong += (0 if is_correct else 1)

    # 3) 오류율 출력
    error_rate = wrong / total if total else 0.0
    print("\n=== 요약 ===")
    print(f"총 개수: {total}마리")
    print(f"틀린 개수: {wrong}마리")
    print(f"오류율: {error_rate*100:.2f}%")

if __name__ == "__main__":
    main()
