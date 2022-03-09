#1. 뷰(View) - 원소의 수를 유지하면서 텐서의 크기 변경. 매우 중요함!!

import numpy as np
import torch

t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)

ft.shape #torch.Size([2, 2, 3]), 데이터 갯수 * 행수 * 열수

#3차원 텐서에서 2차원 텐서로 변경
ft.view([-1,3]) #ft 텐서를 (?,3)의 크기로 변경
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10., 11.]])
ft.view([-1,3]).shape # 4*3

# view는 기본적으로 변경 전과 변경 후의 텐서 안의 원소의 개수가 유지되어야 합니다.
# 파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추합니다.

#3차원 텐서의 크기 변경
print(ft.view([-1, 1, 3]))
# tensor([[[ 0.,  1.,  2.]],
#         [[ 3.,  4.,  5.]],
#         [[ 6.,  7.,  8.]],
#         [[ 9., 10., 11.]]])
print(ft.view([-1, 1, 3]).shape) #torch.Size([4, 1, 3])

#----------------------------------------------------------------
#2. 스퀴즈(Squeeze) - 1인 차원을 제거한다.
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.squeeze())
print(ft.squeeze().shape)
# tensor([0., 1., 2.])
# torch.Size([3])

#언스퀴즈(Unsqueeze) - 압축해제, 특정 위치에 1인 차원을 추가한다.
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
# torch.Size([3])

print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
print(ft.unsqueeze(0).shape)
# tensor([[0., 1., 2.]])
# torch.Size([1, 3])

#위와 같음
print(ft.view(1, -1)) #tensor([[0., 1., 2.]])
print(ft.view(1, -1).shape) #torch.Size([1, 3])


print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

#위와 동일
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)
# view(), squeeze(), unsqueeze()는 텐서의 원소 수를 그대로 유지하면서 모양과 차원을 조절합니다.

#----------------------------------------------------------------
#3.타입 캐스팅(Type Casting)
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())

#----------------------------------------------------------------
#4. 텐서 병합
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
x.shape #torch.Size([2, 2])

print(torch.cat([x, y], dim=0)) #dim=0은 첫번째 차원 (행방향 결합)
print(torch.cat([x, y], dim=0).shape) #torch.Size([4, 2])

print(torch.cat([x, y], dim=1)) #dim=1은 두번째 차원 (열방향 결합)
print(torch.cat([x, y], dim=1).shape) #torch.Size([2, 4])

#스택킹(Stacking)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

#위와 같음
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

#----------------------------------------------------------------
#5) ones_like와 zeros_like - 0으로 채워진 텐서와 1로 채워진 텐서
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채우기
print(torch.zeros_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기

#-----------------------------------------------------------
#6) In-place Operation (덮어쓰기 연산)
x = torch.FloatTensor([[1, 2], 
                       [3, 4]])
print(x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
# tensor([[2., 4.],
#         [6., 8.]])
x*2 #위와 같음
print(x) # 기존의 값 출력


print(x.mul_(2.)) #기본 x의 값에 덮어쓰기됨
print(x)
