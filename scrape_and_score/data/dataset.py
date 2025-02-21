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
      
      self.X = torch.tensor(df.drop(columns=['fantasy_points']).values, dtype=torch.float32)
      self.y = torch.tensor(df["fantasy_points"].values, dtype=torch.float32)
   
   
   def __getitem__(self, idx):
      """Retrieve item (tuple) containing feauters & expected value 

      Args:
          idx (_type_): _description_

      Returns:
          _type_: _description_
      """
      # account for batching by DataLoader 
      if torch.is_tensor(idx):
         idx = idx.tolist() 
      
      return self.X[idx], self.y[idx]
      
   
   
   def __len__(self):
      return len(self.df)
      