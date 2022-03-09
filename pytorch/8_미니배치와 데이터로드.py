#1. 미니 배치와 배치 크기(Mini Batch and Batch Size)
# 그렇기 때문에 전체 데이터를 더 작은 단위로 나누어서 해당 단위로 학습하는 개념이 나오게 되었습니다.
# 이 단위를 미니 배치(Mini Batch)라고 합니다.

# 에포크(Epoch)는 전체 훈련 데이터가 학습에 한 번 사용된 주기를 말한다고 언급한 바 있습니다.
# 배치사이즈 : 미니 배치의 크기
# 배치 크기는 보통 2의 제곱수를 사용합니다. 
# ex) 2, 4, 8, 16, 32, 64... 그 이유는 CPU와 GPU의 메모리가 2의 배수이므로 배치크기가 2의 제곱수일 경우에 데이터 송수신의 효율을 높일 수 있다고 합니다.

# 2. 이터레이션(Iteration)
# 이터레이션은 한 번의 에포크 내에서 이루어지는 매개변수인 가중치 와 의 업데이트 횟수입니다. 
# 전체 데이터가 2,000일 때 배치 크기를 200으로 한다면 이터레이션의 수는 총 10개입니다. 
# 이는 한 번의 에포크 당 매개변수 업데이트가 10번 이루어짐을 의미합니다.

# 3. 데이터 로드하기(Data Load)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

#ensorDataset은 기본적으로 텐서를 입력으로 받습니다. 텐서 형태로 데이터를 정의합니다.
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)

#? 데이터 로더 역할                        
# 데이터로더는 기본적으로 2개의 인자를 입력받는다. 
# 하나는 데이터셋, 미니 배치의 크기입니다. 
# 이때 미니 배치의 크기는 통상적으로 2의 배수를 사용합니다. (ex) 64, 128, 256...) 
# 그리고 추가적으로 많이 사용되는 인자로 shuffle이 있습니다. shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿉니다.

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20

for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    print(batch_idx)
    print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader), #len(dataloader) : 이터레이션 값
        cost.item()
        ))
    
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 

