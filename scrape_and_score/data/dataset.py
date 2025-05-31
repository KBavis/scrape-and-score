import torch
from torch.utils.data import Dataset
from .nn_preprocess import scale_and_transform 



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
        self.X = torch.from_numpy(scale_and_transform(df)).float()
        self.y = torch.from_numpy(df["fantasy_points"].values).float()

    def __getitem__(self, idx):
        """Retrieve item (tuple) containing feauters & expected value

        Args:
            idx (torch.tensor): index to retireve item from

        Returns:
            torch.tensor: relevant item
        """
        # account for batching by DataLoader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.df)
