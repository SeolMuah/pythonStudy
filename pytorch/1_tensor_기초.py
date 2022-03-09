import torch

# 2차원으로 구성된 값을 행렬(Matrix)라고 합니다. 
# 그리고 3차원이 되면 우리는 텐서(Tensor)라고 부릅니다. 
# 사실 우리는 3차원의 세상에 살고 있으므로, 4차원 이상부터는 머리로 생각하기는 어렵습니다. 
# 4차원은 3차원의 텐서를 위로 쌓아 올린 모습으로 상상해보겠습니다.
# 1차원 벡터나 2차원인 행렬도 텐서라고 표현하기도 합니다.

#1) 1D Tensor
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print(t.dim())  # rank. 즉, 차원
print(t.shape)  # shape
print(t.size()) # shape

print(t[0], t[1], t[-1])  # 인덱스로 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱

#2) 2D Tensor
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)
print(t.dim())  # rank. 즉, 차원
print(t.size()) # shape

print(t[:, 1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원의 첫번째 것만 가져온다.
print(t[:, 1].size()) # ↑ 위의 경우의 크기


#3) 브로드캐스팅
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] -> [3, 3]
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# # 브로드캐스팅 과정에서 실제로 두 텐서가 어떻게 변경되는지 보겠습니다.
# [1, 2]
# ==> [[1, 2],
#      [1, 2]]
# [3]
# [4]
# ==> [[3, 3],
#      [4, 4]]

#행렬 곱셈과 곱셈의 차이
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

#평균
t = torch.FloatTensor([1, 2])
print(t.mean())

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean())
print(t.mean(dim=0)) #행방향
print(t.mean(dim=1)) #열방향
print(t.mean(dim=-1)) #열방향

#덧셈
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.sum())
print(t.sum(dim=0)) #행방향
print(t.sum(dim=1)) #열방향
print(t.sum(dim=-1)) #열방향

#최댓값, 최댓값 갖는 인덱스
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.max()) # Returns one value: max
print(t.max(dim=0)) # Returns two values: max and argmax
print(t.max(dim=1)) # Returns two values: max and argmax

print(t.max(dim=0)[0]) #max
print(t.max(dim=0)[1]) #argmax