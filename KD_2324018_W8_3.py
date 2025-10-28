# w = [-1.5, 1.0, 1.0]
# [-3, 2, 2] 도 가능

# 퍼셉트론 파라미터
w = [-1.5, 1.0, 1.0]   # w0(바이어스), w1, w2

# 활성함수
def step(z):
    return 1 if z > 0 else 0 # 계단 함수

# 퍼셉트론 함수
def perceptron(x1, x2, w):
    z = w[0] + w[1]*x1 + w[2]*x2      # 활성함수 전(가중합)
    o = step(z)                       # 활성함수 후(출력)
    return z, o                       # 전, 후 값 반환

# 검증용 벡터
inputs = [(0,0), (0,1), (1,0), (1,1)]

print("파라미터 값")
print(f"w0(바이어스)={w[0]}, w1={w[1]}, w2={w[2]}\n")

print("AND 퍼셉트론 검증")
for x1, x2 in inputs:
    z, o = perceptron(x1, x2, w)
    print(f"입력 x=[{x1}, {x2}] -> z(전)={z} , o(후)={o}")
