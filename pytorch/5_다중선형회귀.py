import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#H(x) = w1*x1 + w2*x2 + w3*x3 + b

# 훈련 데이터
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w와 편향 b 초기화
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad() #이전 step에서 각 layer 별로 계산된 gradient 값을 모두 0으로 초기화 시키는 작업입니다.
    cost.backward() #각 layer의 파라미터에 대하여 back-propagation을 통해 gradient를 계산합니다.
    optimizer.step() #각 layer의 파라미터와 같이 저장된 gradient 값을 이용하여 파라미터를 업데이트 합니다.

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
#----------------------------------------------------------------
#벡터와 행렬 연산으로 바꾸기

# 위의 코드를 개선할 수 있는 부분이 있습니다. 이번에는 의 개수가 3개였으니까 x1_train, x2_train, x3_train와 w1, w2, w3를 일일히 선언해주었습니다. 
# 그런데 의 개수가 1,000개라고 가정해봅시다. 위와 같은 방식을 고수할 경우 x_train1 ~ x_train1000을 전부 선언하고, w1 ~ w1000을 전부 선언해야 합니다. 
# 다시 말해 와  변수 선언만 총 합 2,000개를 해야합니다. 
# 또한 가설을 선언하는 부분에서도 마찬가지로 x_train과 w의 곱셈이 이루어지는 항을 1,000개를 작성해야 합니다. 
# 이는 굉장히 비효율적입니다.
#행렬의 곱셈 과정에서 이루어지는 벡터 연산을 벡터의 내적(Dot Product)이라고 합니다.

#행렬 연산을 고려하여 파이토치로 구현하기
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  #데이터 5개 * 특성 3개

y_train  =  torch.FloatTensor([[152],  
                               [185],  
                               [180],  
                               [196],  
                               [142]]) #각 데이터에 대한 정답 5개

print(x_train.shape) #torch.Size([5, 3])
print(y_train.shape) #torch.Size([5, 1])

# 가중치와 편향 선언
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
W, b

#tensor([[0.],
        # [0.],
        # [0.]], requires_grad=True), tensor([0.], requires_grad=True))
#----------------------------------------------------------------        
#전체 코드
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print(hypothesis)
    print(hypothesis.squeeze())
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item() # detach()는 이 연산 기록으로 부터 분리한 tensor을 반환하는 method이다.
    ))
        
        
