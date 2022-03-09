# 경사 하강법 코드를 보고있으면 requires_grad=True, backward() 등이 나옵니다. 
# 이는 파이토치에서 제공하고 있는 자동 미분(Autograd) 기능을 수행하고 있는 것입니다.
# 파이토치의 학습 과정을 보다 더 잘 이해하기 위해서 자동 미분에 대해서 이해해봅시다.

#y = 2*w^2 + 5 라는 가설을 세우고
#w에 대해 미분

import torch
w = torch.tensor(2.0,requires_grad=True)

z = 2*w**2 + 5
z.backward()

print('수식을 w로 미분한 값 : {}'.format(w.grad)) #수식을 w로 미분한 값 : 8.0

