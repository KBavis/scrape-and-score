import torch
from torch import nn
from torch.utils.data import DataLoader

def optimization_loop(train_data_loader: torch.utils.data.DataLoader, test_data_loader: torch.utils.data.DataLoader, model: nn.Module):
   """Generate optimization loop used to train and test our neural netwokr 

   Args:
      train_data_loader (torch.utils.data.DataLoader): training data loader 
      test_data_loader (torch.utils.data.DataLoader): testing data loader 
   """

   loss_fn = nn.MSELoss() 
   learning_rate = 0.01
   
   optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
   epochs = 250
   
   for training_iteration in range(epochs):
      print(f"Starting Epoch {training_iteration + 1}\n--------------------------")
      train_loop(train_data_loader, model, loss_fn, optimizer)
      test_loop(test_data_loader, model, loss_fn)
   
   print("\nDone with optimization loop")
   
   
def train_loop(dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_fn: nn.MSELoss, optimizer: torch.optim.SGD): 
   """Loop for training our neural network 

   Args:
       dataloader (torch.utils.data.DataLoader): our dataloader containing training data 
       model (nn.Module): our neural network model 
       loss_fn (nn.MSELoss): loss function to determine performance of model 
       optimizer (torch.optim.SGD): optimizer to perform gradient descent (optimize our weights/biases)
   """
   size = len(dataloader.dataset)
   model.train() 
   
   for batch, (X, y) in enumerate(dataloader): 
      pred = model(X).squeeze(1)
      
      loss = loss_fn(pred, y)
      
      #back propogation
      loss.backward() 
      optimizer.step() 
      optimizer.zero_grad() 
      
      if batch % 10 == 0: 
         loss, current = loss.item(), batch * batch * 64 + len(X)
         print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_fn: nn.MSELoss, tolerance: float = 3.0): 
   """Loop for testing our neural network 

   Args:
       dataloader (torch.utils.data.DataLoader): _description_
       model (nn.Module): neural network model 
       loss_fn (nn.MSELoss): loss function we are utilizing 
   """
   
   model.eval() 
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   test_loss = 0
   correct = 0
   
   
   with torch.no_grad(): # no need for gradients int testing 
      for X, y in dataloader:
         pred = model(X).squeeze(1)
         test_loss += loss_fn(pred, y).item() 
         correct += torch.sum(torch.abs(pred - y) <= tolerance).item() 
   
   
   test_loss /= num_batches 
   accuracy = 100 * (correct / size) 

    # Print final metrics
   print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
      
   