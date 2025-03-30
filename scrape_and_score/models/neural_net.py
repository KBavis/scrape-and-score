import torch
from torch import nn

class NeuralNetwork(nn.Module):
   def __init__(self, input_dim: int):
      super().__init__()

      self.linear_relu_stack = nn.Sequential(
         nn.Linear(input_dim, 1024),    
         nn.ReLU(),
         nn.BatchNorm1d(1024),
         nn.Dropout(0.6),         
         nn.Linear(1024, 512),    
         nn.ReLU(),
         nn.BatchNorm1d(512),
         nn.Dropout(0.5),
         nn.Linear(512, 256),    
         nn.ReLU(),
         nn.BatchNorm1d(256),
         nn.Dropout(0.4),
         nn.Linear(256, 128),    
         nn.ReLU(),
         nn.BatchNorm1d(128),
         nn.Linear(128, 1)        
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
   
