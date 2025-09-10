# 규칙 임계값 (조건 기준)
BODY_rule = 92   # 몸길이 기준값
TAIL_rule = 9    # 꼬리길이 기준값

# 입출력 파일 이름
INPUT_FILE  = "input_data.txt"      # 입력 데이터 파일
OUTPUT_FILE = "output_result.txt"   # 결과 저장 파일

def classify(body, tail):
    """몸/꼬리 길이에 따라 어종 분류"""
    # 몸이 짧고 꼬리가 길면 연어(salmon), 아니면 농어(seabass)
    return "salmon" if body < BODY_rule and tail > TAIL_rule else "seabass"

def main():
    # 입력 파일 읽기 + 출력 파일 쓰기
    with (open(INPUT_FILE, "r", encoding="utf-8") as fin,
          open(OUTPUT_FILE, "w", encoding="utf-8") as fout):

        # 한 줄씩 읽어서 처리
        for i, line in enumerate(fin, start=1):
            body, tail = map(int, line.split())  # 정수 변환 (몸/꼬리 길이)

            # 규칙에 따라 분류
            result = classify(body, tail)
            out = f"Fish{i}: body={body}, tail={tail} → {result}"

            # 화면 출력 + 파일 저장
            print(out)
            fout.write(out + "\n")

    print(f"\n 결과가 '{OUTPUT_FILE}' 로 저장되었습니다.")

if __name__ == "__main__":
    main()
