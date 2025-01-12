from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns 


class LinReg:
   def __init__(self, qb_data, rb_data, wr_data, te_data):
      self.qb_data = qb_data 
      self.rb_data = rb_data
      self.wr_data = wr_data 
      self.te_data = te_data
      
      self.qb_inputs_scaled = self.scale_inputs(qb_data)   
      self.rb_inputs_scaled = self.scale_inputs(rb_data)   
      self.wr_inputs_scaled = self.scale_inputs(wr_data)   
      self.te_inputs_scaled = self.scale_inputs(te_data)   
      
      self.qb_targets = qb_data['log_fantasy_points']
      self.rb_targets = rb_data['log_fantasy_points']
      self.wr_targets = wr_data['log_fantasy_points']
      self.te_targets = te_data['log_fantasy_points']
      
      self.qb_regression = None
      self.rb_regression = None
      self.wr_regression = None
      self.te_regression = None

   '''
   Functionality to scale inputs 
   
   Args:
      df (pd.DataFrmae): data frame to scale inputs for 
   
   Returns
      inputs (pd.DataFrame): inputs from data frame
   ''' 
   def scale_inputs(self, df: pd.DataFrame): 
      inputs = df[(df['log_avg_fantasy_points'].notnull() | df['log_ratio_rank'].notnull())]
      scaler = StandardScaler() 
      scaler.fit(inputs)
      
      return scaler.transform(inputs)
      
   '''
   Functionality to split scale inputs into training and testing data 
   
   Args:
      None 
   
   Returns: 
      split_data (dict): dictionary containing split data for each position
   '''
   def create_training_and_testing_split(self):
      qb_x_train, qb_x_test, qb_y_train, qb_y_test = train_test_split(self.qb_inputs_scaled, self.qb_targets, test_size=0.2, random_state=365)
      rb_x_train, rb_x_test, rb_y_train, rb_y_test = train_test_split(self.rb_inputs_scaled, self.rb_targets, test_size=0.2, random_state=365)
      wr_x_train, wr_x_test, wr_y_train, wr_y_test = train_test_split(self.wr_inputs_scaled, self.wr_targets, test_size=0.2, random_state=365)
      te_x_train, te_x_test, te_y_train, te_y_test = train_test_split(self.te_inputs_scaled, self.te_targets, test_size=0.2, random_state=365)
      
      return {
         "QB": {
            "x_train": qb_x_train,
            "x_test": qb_x_test,
            "y_train": qb_y_train,
            "y_test": qb_y_test
         },
         "RB": {
            "x_train": rb_x_train,
            "x_test": rb_x_test,
            "y_train": rb_y_train,
            "y_test": rb_y_test
         },
         "WR": {
            "x_train": wr_x_train,
            "x_test": wr_x_test,
            "y_train": wr_y_train,
            "y_test": wr_y_test
         },
         "TE": {
            "x_train": te_x_train,
            "x_test": te_x_test,
            "y_train": te_y_train,
            "y_test": te_y_test
         }
      }
   
   
   def create_regressions(self):
      split_data = self.create_training_and_testing_split()
      
      self.qb_regression = LinearRegression()
      qb_x_train = split_data['QB']['x_train']
      qb_y_train = split_data['QB']['y_train']
      print(qb_x_train)
      print(qb_y_train)
      
      self.qb_regression.fit(qb_x_train, qb_y_train)
      qb_y_hat = self.qb_regression.predict(qb_x_train)
      
      #TODO: Repeat for other positions
      
      
      #TODO: Create seperate scatter plot creation function
      plt.scatter(qb_y_train, qb_y_hat)
      
      
      plt.xlabel('Targets (y_train)', size=18)
      plt.ylabel('Predictions (y_hat)', size=18)
      
      # plt.xlim(6, 13)
      # plt.ylim(6,13)
      
      plt.savefig('linregqb.pdf')
      
      sns.displot(qb_y_train - qb_y_hat)
      plt.savefig('residuals.pdf')