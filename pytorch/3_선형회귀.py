# 데이터에 대한 이해(Data Definition)
# 학습할 데이터에 대해서 알아봅니다.

# 가설(Hypothesis) 수립
# 가설을 수립하는 방법에 대해서 알아봅니다.

# 손실 계산하기(Compute loss)
# 학습 데이터를 이용해서 연속적으로 모델을 개선시키는데 이 때 손실(loss)를 이용합니다.

# 경사 하강법(Gradient Descent)
# 학습을 위한 핵심 알고리즘인 경사 하강법(Gradient Descent)에 대해서 이해합니다.

#--------------------------------------------------------
# 1.가설(Hypothesis) 수립
#y = Wx + b
#H(x) = Wx + b #H는 가설의 약자

# 2.손실 계산하기(Compute loss)
# 2-1) 비용함수 이해
# 비용 함수(cost function) = 손실 함수(loss function) = 오차 함수(error function) = 목적 함수(objective function)

# 예측 모델 y = 13x + 1
# hours(x)	2	3	4	5
# 실제값	 25	50	42	61
# 예측값	 27	40	53	66
# 오차	    -2	10	-9	-5

# 평균 제곱 오차(Mean Squared Error, MSE)
totErr = 2**2 + 10**2 + 9**2 + 5**2
avgErr = totErr / 4
totErr, avgErr #(210, 52.5)

# 3. 옵티마이저 - 경사 하강법(Gradient Descent)
# 최적화 알고리즘이라고도 부릅니다. 
# 그리고 이 옵티마이저 알고리즘을 통해 적절한 와 를 찾아내는 과정을 머신 러닝에서 학습(training)이라고 부릅니다. 
# 여기서는 가장 기본적인 옵티마이저 알고리즘인 경사 하강법(Gradient Descent)에 대해서 배웁니다.

#비용함수를 미분하여 현재 W에서의 접선의 기울기를 구하고, 접선의 기울기가 낮은 방향으로 W의 값을 변경하는 작업을 반복
#기울기 = a Cost(W) / a W
#W = W - l*기울기 (l : 학습률)

#--------------------------------------------------------
#1. 기본세팅
from cmath import cos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#2. 데이터 로드
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#3. 가중치와 편향의 초기화
# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
W = torch.zeros(1, requires_grad=True)  #requires_grad=True 는 autograd(자동미분 에 모든 연산(operation)들을 추적해야 한다고 알려줍니다.
# 가중치 W를 출력
print(W) 

b = torch.zeros(1, requires_grad=True)
print(b)
#학습전 모델 : y=0x+0 

#4. 가설세우기
hypo = x_train * W + b
hypo

#5. 비용함수 선언
cost = torch.mean((hypo - y_train)**2)
cost #tensor(18.6667, grad_fn=<MeanBackward0>)

#6. 경사하강법구현
optimizer = optim.SGD([W,b], lr=0.01)
# gradient를 0으로 초기화
optimizer.zero_grad() 
# 비용 함수를 미분하여 gradient 계산
cost.backward() 
# W와 b를 업데이트
optimizer.step()

#---------------------------------------------------------------
#전체 코드

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 2000 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b 

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
        
#----------------------------------------------------------------
#optimizer.zero_grad()가 필요한 이유
#파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있습니다. 예를 들어봅시다.
import torch
w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
  z = 2*w
  z.backward()
  print('수식을 w로 미분한 값 : {}'.format(w.grad))