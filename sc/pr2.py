import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,16), nn.ReLU(),
            nn.Linear(16,16), nn.ReLU(),
            nn.Linear(16,2)
        )
    def forward(self,x):
        return self.net(x)
# Data
x = torch.randn(100,3)
y = torch.randint(0,2,(100,))
# Setup
model = MLP()
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.01)
losses=[]
for epoch in range(200):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")
# Plot
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
# Accuracy
with torch.no_grad():
    acc = (model(x).argmax(1)==y).float().mean()
    print("Accuracy:", acc.item()*100)
