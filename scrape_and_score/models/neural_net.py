import torch
from torch import nn

class NeuralNetwork(nn.Module):
   def __init__(self, input_dim: int, position: str):
      super().__init__()

      # mapping of position to specific layers 
      positional_layers = {
            'QB': self.get_qb_linear_relu_stack(input_dim), 
            'RB': self.get_rb_linear_relu_stack(input_dim), 
            'WR': self.get_wr_linear_relu_stack(input_dim), 
            'TE': self.get_te_linear_relu_stack(input_dim)
      }

      self.linear_relu_stack = positional_layers[position]
  
   
   def forward(self, x: torch.Tensor):
      """Execute the forward pass of our neural network 

      Args:
          x (torch.Tensor): tensor to pass through neural network 

      Returns:
          torch.Tensor : output of forward pass
      """
      logits = self.linear_relu_stack(x)
      return logits
   

   def get_qb_linear_relu_stack(self, input_dim: int):
      """
      Retrieve layers for a QB Neural Network model
      """

      return nn.Sequential(
         nn.Linear(input_dim, 256),     
         nn.ReLU(),
         nn.BatchNorm1d(256),
         nn.Dropout(0.7),               
         nn.Linear(256, 128),
         nn.ReLU(),
         nn.BatchNorm1d(128),
         nn.Dropout(0.6),
         nn.Linear(128, 64),           
         nn.ReLU(),
         nn.Linear(64, 1)
      )

   def get_wr_linear_relu_stack(self, input_dim: int):
      """
      Retrieve layers for a WR Neural Network model
      """

      return nn.Sequential(
         nn.Linear(input_dim, 1024),
         nn.ReLU(),
         nn.BatchNorm1d(1024),
         nn.Dropout(0.5),              
         nn.Linear(1024, 512),
         nn.ReLU(),
         nn.BatchNorm1d(512),
         nn.Dropout(0.4),
         nn.Linear(512, 256),
         nn.ReLU(),
         nn.BatchNorm1d(256),
         nn.Dropout(0.3),
         nn.Linear(256, 1)
      )


   def get_te_linear_relu_stack(self, input_dim: int):
      """
      Retrieve layers for a TE Neural Network model
      
      """

      return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
      )
      

   def get_rb_linear_relu_stack(self, input_dim: int):
      """
      Retrieve layers for a RB Neural Network model
      
      """
      return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
      )
   
