import torch as tc
import torch.nn as nn
import torch.nn.functional as F


class FCModel(nn.Module):
  def __init__(self, input_size=2048):
    super(FCModel, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.fc2 = nn.Linear(128, 1)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x
