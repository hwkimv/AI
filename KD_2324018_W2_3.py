# 규칙 임계값
BODY_THRESHOLD = 92
TAIL_THRESHOLD = 9

# 입출력 파일
INPUT_FILE  = "input_data.txt"
OUTPUT_FILE = "output_result.txt"

def classify(body, tail):
    """몸/꼬리 길이에 따라 어종 분류"""
    return "salmon" \
        if body < BODY_THRESHOLD and tail > TAIL_THRESHOLD else "seabass"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
            open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin, start=1):
            body, tail = map(int, line.split())

            # 규칙 기반으로 어종 분류
            result = classify(body, tail)
            out = f"Fish{i}: body={body}, tail={tail} → {result}"
            print(out)
            fout.write(out + "\n")

    print(f"\n 결과가 '{OUTPUT_FILE}' 로 저장되었습니다.")

if __name__ == "__main__":
    main()
