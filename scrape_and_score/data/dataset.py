import torch 
from torch.utils.data import Dataset

class FantasyDataset(Dataset): 

   """
      Args:
         df (pd.DataFrame): data frame containing all relevant indepdent/dependent variables 
         transform (callable, optional): optional transform to be applied
   """
   def __init__(self, df, transform=None):
      super().__init__()
      self.df = df
      self.transform = transform
   
   
   def __getitem__(self, idx):
      # account for batching by DataLoader 
      if torch.is_tensor(idx):
         idx = idx.tolist() 
      
      #TODO: Implement me 
      return None
      
   
   
   def __len__(self):
      return len(self.df)
      