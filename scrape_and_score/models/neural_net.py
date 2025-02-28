import torch
from torch import nn

class NeuralNetwork(nn.Module):
   def __init__(self):
      super().__init__()
      
      self.linear_relu_stack = nn.Sequential(
         nn.Linear(46, 32), #TODO: Update this if the number of columns changes 
         nn.ReLU(), 
         nn.Linear(32, 1)
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
   
