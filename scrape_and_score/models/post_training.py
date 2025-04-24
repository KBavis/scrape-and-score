import torch
import shap
import pandas as pd
import numpy as np
import logging
from datetime import datetime


def feature_importance(model: torch.nn.Module, training_data_set: torch.utils.data.Dataset, position: str): 
    """
    Determine which features are most signficant in terms of the predictive value of i

    Args:
        model (torch.nn.Module): neural network model
        training_data_set (torch.utils.data.Dataset): the data set utilized for training
        position (str): the specific position this model corresponds to 
    """
    logging.info(f'Post Training: Determining feature signficance of our {position} model')


    background = training_data_set.X[:100]
    samples_to_explain = training_data_set.X[:200] 

    explainer = shap.DeepExplainer(model, background)

    shap_values = explainer.shap_values(samples_to_explain)

    # extract input columns 
    columns = list(training_data_set.df.columns)
    inputs = [col for col in columns if col != "fantasy_points"]

    # adjust dimension of shap_values 
    shap_values = np.squeeze(shap_values)

    feature_importance = pd.DataFrame({
        'Feature': inputs,
        'SHAP_Importance': np.abs(shap_values).mean(axis = 0)
    }).sort_values('SHAP_Importance', ascending=False)

    print(f"{position} Model Most Important Feautres: \n{feature_importance.head}")

    # save feature significance table 
    save_model_feature_significance_table(feature_importance, position)


def save_model_feature_significance_table(df: pd.DataFrame, position: str):
    """
    Helper function to save model feature significances 

    Args:
        df (pd.DataFrame): data frame containing training data 
        position (str): position we are determing features signficance for
    """
    # Generate a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data/shap_analysis/{position}_feature_summary_table_{timestamp}.csv"

    # Save DataFrame as CSV
    df.to_csv(file_name, index=False)
    
    logging.info(f"Saved feature summary table as {file_name}")




