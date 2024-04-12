import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset, DataLoader
from model import *  # 모델.py에서 정의된 모델을 불러옴
from utils import MusicNetSpectrogramDataset

batch_size = 32
num_epochs = 50
learning_rate = 0.001

dataset = MusicNetSpectrogramDataset(root_dir=root_dir, files=files)
train_data = DataLoader(dataset, batch_size=1, shuffle=True)


# 모델, 손실 함수, 옵티마이저 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NSynthEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
def train():
    model.train()
    for epoch in range(num_epochs):
        for spec, sample_rate in train_data:
            spec, sample_rate = train_data.to(device)

            # Forward pass
            outputs = model(train_data)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 모델 저장 (여기서는 각 epoch 마다 저장)
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        print(f'Model saved: model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()
