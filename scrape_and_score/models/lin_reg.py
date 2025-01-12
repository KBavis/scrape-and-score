from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns 
import logging 
import numpy as np
import os


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
      
      self.split_data = None

   '''
   Functionality to scale inputs 
   
   Args:
      df (pd.DataFrmae): data frame to scale inputs for 
   
   Returns
      inputs (pd.DataFrame): inputs from data frame
   ''' 
   def scale_inputs(self, df: pd.DataFrame): 
      inputs = df.drop(['log_fantasy_points'], axis=1)
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
      
      
      # reset indices
      qb_y_test = qb_y_test.reset_index(drop=True)
      qb_y_train = qb_y_train.reset_index(drop=True)
      rb_y_train = rb_y_train.reset_index(drop=True)
      rb_y_test = rb_y_test.reset_index(drop=True)
      wr_y_test = wr_y_test.reset_index(drop=True)
      wr_y_train = wr_y_train.reset_index(drop=True)
      te_y_test = te_y_test.reset_index(drop=True)
      te_y_train = te_y_train.reset_index(drop=True)
      
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
   
   
   '''
   Functionality to create all multiple linear regression models for each NFL relevant position 
   
   Args:
      None
   
   Returns:
      None
   '''
   def create_regressions(self):
      split_data = self.create_training_and_testing_split()
      self.split_data = split_data
      
      positions = ['QB', 'RB', 'WR', 'TE']
      
      regression_model_mapping = {}
      for position in positions: 
         regression_model_mapping[position] = self.create_position_regression(split_data[position]['x_train'], split_data[position]['y_train'], position)
      
      self.qb_regression = regression_model_mapping['QB']
      self.rb_regression = regression_model_mapping['RB']
      self.te_regression = regression_model_mapping['TE']
      self.wr_regression = regression_model_mapping['WR']
      
      self.log_regression_metrics()
   
   
   '''
   Log out relevant regression bias/weights associated with model
   
   Args:
      None
   
   Returns:
      None
   '''
   def log_regression_metrics(self):
      # QB 
      print('\n\n')
      logging.info('QB Regression Model Metrics')
      logging.info(f'Bias Of Regression (Y-Intercept): {self.qb_regression.intercept_}')
      
      inputs = self.qb_data.drop(['log_fantasy_points'], axis=1)
      reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
      reg_summary['Weights'] = self.qb_regression.coef_
      logging.info(f'{reg_summary}')
      
      # RB 
      print('\n\n')
      logging.info('RB Regression Model Metrics')
      logging.info(f'Bias Of Regression (Y-Intercept): {self.rb_regression.intercept_}')
      
      inputs = self.rb_data.drop(['log_fantasy_points'], axis=1)
      reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
      reg_summary['Weights'] = self.rb_regression.coef_
      logging.info(f'{reg_summary}')
      
      # WR 
      print('\n\n')
      logging.info('WR Regression Model Metrics')
      logging.info(f'Bias Of Regression (Y-Intercept): {self.wr_regression.intercept_}')
      
      inputs = self.wr_data.drop(['log_fantasy_points'], axis=1)
      reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
      reg_summary['Weights'] = self.wr_regression.coef_
      logging.info(f'{reg_summary}')
      
      # TE
      print('\n\n')
      logging.info('TE Regression Model Metrics')
      logging.info(f'Bias Of Regression (Y-Intercept): {self.te_regression.intercept_}')
      
      inputs = self.te_data.drop(['log_fantasy_points'], axis=1)
      reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
      reg_summary['Weights'] = self.te_regression.coef_
      logging.info(f'{reg_summary}')
      
   
      '''
   Helper function for creating the position specific regression models
   
   Args:
      x_train (pandas.DataFrame): data frame containing x training data 
      y_train (pandas.DataFrame): data frame containing y training data 
      position (str): position this model is created for 
   
   Returns:
      regression_model (sklearn.linear_model.LinearRegression): linear regression model used for predictions
   '''
   def create_position_regression(self, x_train, y_train, position):
      regression = LinearRegression() 
      
      regression.fit(x_train, y_train) # train model 
      y_hat = regression.predict(x_train) # test model against trained data
      
      # create relevant graphs
      self.create_prediction_scatter_plot(y_train, y_hat, f"{position}_training_predictions") 
      df = y_train - y_hat
      self.create_residuals_dist(df, f"{position}_residuals") 
      
      logging.info(f'R-squared for {position} multiple linear regression model: {regression.score(x_train, y_train)}')
      return regression
   
   '''
   Create distribution plot of residuals (predicted - expected)
   
   Args:
      df (pd.DataFrame): data frame containing residuals
      file_name (str): file name 
   
   Returns:
      None
   '''
   def create_residuals_dist(self, df: pd.DataFrame, file_name: str):
      relative_dir = "./data/distributions"
      pdf = f"{file_name}.pdf"
      os.makedirs(relative_dir, exist_ok=True)
      file_path = os.path.join(relative_dir, pdf)
      
      sns.displot(df)
      
      plt.savefig(file_path)
      plt.figure() # create new figure

   
   '''
   Functionality to create & save scatter plot for model predictions vs actual values 
   
   Args:
      y (pd.DataFrame) : data frame containing actual values 
      y_hat (pd.DataFrame): data frame containing predicted values 
      file_name (str): file name to save scatter plot as
   
   Returns:
      None
   '''
   def create_prediction_scatter_plot(self, y, y_hat, file_name):
      relative_dir = "./data/scatter"
      pdf = f"{file_name}.pdf"
      os.makedirs(relative_dir, exist_ok=True)
      file_path = os.path.join(relative_dir, pdf)
      
      plt.scatter(y, y_hat)  
      plt.xlabel('Targets (y)', size=18)
      plt.ylabel('Predictions (y_hat)', size=18)
      plt.xlim(0,5)
      plt.ylim(0,10)
      plt.savefig(file_path)
      plt.figure()


   '''
   Test each of our regressions against our testing data 

   Args:
      None

   Returns:
      None
   '''
   def test_regressions(self):
      qb_y_test = self.split_data['QB']['y_test']
      rb_y_test = self.split_data['RB']['y_test']
      wr_y_test = self.split_data['WR']['y_test']
      te_y_test = self.split_data['TE']['y_test']
      
      qb_y_hat_test = self.qb_regression.predict(self.split_data['QB']['x_test'])
      te_y_hat_test = self.te_regression.predict(self.split_data['TE']['x_test'])
      wr_y_hat_test = self.wr_regression.predict(self.split_data['WR']['x_test'])
      rb_y_hat_test = self.rb_regression.predict(self.split_data['RB']['x_test'])
   
      self.create_prediction_scatter_plot(qb_y_test, qb_y_hat_test, "QB_testing_predictions")
      self.create_prediction_scatter_plot(rb_y_test, rb_y_hat_test, "RB_testing_predictions")
      self.create_prediction_scatter_plot(wr_y_test, wr_y_hat_test, "WR_testing_predictions")
      self.create_prediction_scatter_plot(te_y_test, te_y_hat_test, "TE_testing_predictions")
      
      df_qb_predictions = pd.DataFrame(np.exp(qb_y_hat_test), columns=['Predictions'])
      df_qb_predictions['Target'] = np.exp(qb_y_test)
      
      df_rb_predictions = pd.DataFrame(np.exp(rb_y_hat_test), columns=['Predictions'])
      df_rb_predictions['Target'] = np.exp(rb_y_test)
      
      df_wr_predictions = pd.DataFrame(np.exp(wr_y_hat_test), columns=['Predictions'])
      df_wr_predictions['Target'] = np.exp(wr_y_test)
      
      df_te_predictions = pd.DataFrame(np.exp(te_y_hat_test), columns=['Predictions'])
      df_te_predictions['Target'] = np.exp(te_y_test)
      
      df_qb_predictions['Residual'] = df_qb_predictions['Target'] - df_qb_predictions['Predictions']
      df_rb_predictions['Residual'] = df_rb_predictions['Target'] - df_rb_predictions['Predictions']
      df_wr_predictions['Residual'] = df_wr_predictions['Target'] - df_wr_predictions['Predictions']
      df_te_predictions['Residual'] = df_te_predictions['Target'] - df_te_predictions['Predictions']
      
      df_qb_predictions['Difference %'] = np.absolute(df_qb_predictions['Residual'] / df_qb_predictions['Target'] * 100)
      df_rb_predictions['Difference %'] = np.absolute(df_rb_predictions['Residual'] / df_rb_predictions['Target'] * 100)
      df_wr_predictions['Difference %'] = np.absolute(df_wr_predictions['Residual'] / df_wr_predictions['Target'] * 100)
      df_te_predictions['Difference %'] = np.absolute(df_te_predictions['Residual'] / df_te_predictions['Target'] * 100)
      
      
      pd.options.display.max_rows = 999 # view all rows 
      pd.set_option('display.float_format', lambda x: '%.2f' % x)

      sorted_qb_df_predictions = df_qb_predictions.sort_values(by=['Difference %'])
      sorted_rb_df_predictions = df_rb_predictions.sort_values(by=['Difference %'])
      sorted_wr_df_predictions= df_wr_predictions.sort_values(by=['Difference %'])
      sorted_te_df_predictions = df_te_predictions.sort_values(by=['Difference %'])
      
      print('\n\nQB Testing Predictions')
      print(sorted_qb_df_predictions)
      
      print('\n\nRB Testing Predictions')
      print(sorted_rb_df_predictions)
      
      print('\n\nWR Testing Predictions')
      print(sorted_wr_df_predictions)
      
      print('\n\nTE Testing Predictions')
      print(sorted_te_df_predictions)