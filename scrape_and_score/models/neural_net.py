import torch
from torch import nn

class NeuralNetwork(nn.Module):
   def __init__(self, input_dim: int):
      super().__init__()
      
      # self.linear_relu_stack = nn.Sequential(
      #    nn.Linear(input_dim, 32), 
      #    nn.ReLU(), 
      #    nn.Linear(32, 1)
      # )

      # self.linear_relu_stack = nn.Sequential(
      #    nn.Linear(45, 64),
      #    nn.ReLU(), 
      #    nn.Linear(64, 32),
      #    nn.ReLU(),
      #    nn.Linear(32, 16),
      #    nn.ReLU(),
      #    nn.Linear(16, 1)
      # )
      # self.linear_relu_stack = nn.Sequential(
      #    nn.Linear(input_dim, 64), 
      #    nn.ReLU(),
      #    nn.Linear(64, 32),        
      #    nn.ReLU(),
      #    nn.Linear(32, 16),       
      #    nn.ReLU(),
      #    nn.Linear(16, 1)   
      # )
      self.linear_relu_stack = nn.Sequential(
         nn.Linear(input_dim, 128),  # Input layer
         nn.ReLU(),
         nn.BatchNorm1d(128),
         nn.Dropout(0.3),
         nn.Linear(128, 64),  # Hidden layer
         nn.ReLU(),
         nn.BatchNorm1d(64),
         nn.Dropout(0.2),
         nn.Linear(64, 1)     # Output layer
      )
   
   def forward(self, x: torch.Tensor):
      """Execute the forward pass of our neural network 

      Args:
          x (torch.Tensor): tensor to pass through neural network 

      Returns:
          torch.Tensor : output of forward pass
      """
      logits = self.linear_relu_stack(x)
      return logits
   
