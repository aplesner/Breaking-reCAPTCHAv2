import pandas as pd
from ultralytics import YOLO
import torch
import numpy as np

"""
This script performs a grid search for hyperparameter tuning of a YOLO model. 
It loads a pre-trained model and a DataFrame of previous results, if available. 
The hyperparameters include epochs, image size, initial learning rate, number of first layers to freeze, 
SGD momentum, label smoothing, and dropout regularization. 
If a combination of hyperparameters has been previously tested, it is skipped. 
The results are stored in a DataFrame for future reference.
"""

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Specify the save directory
save_dir = ''

# Define the grid of hyperparameters to search
epochs_grid = [5, 10, 20]
imgsz_grid = [120]
lr0_grid = [0.01, 0.001, 0.0001]  # initial learning rate
freeze_grid = [np.nan, 5, 10]  # freeze first n layers
momentum_grid = [0.9, 0.95, 0.99]  # SGD momentum
label_smoothing_grid = [0.0, 0.1, 0.2]  # label smoothing
dropout_grid = [0.0, 0.1, 0.2]  # dropout regularization

# Load previous results if they exist
try:
    results_df = pd.read_csv('grid_search_results.csv')
    best_top1_accuracy = results_df['top1_accuracy'].max()
except FileNotFoundError:
    results_df = pd.DataFrame(columns=['epochs', 'imgsz', 'lr0', 'freeze', 'momentum', 'label_smoothing', 'dropout', 'top1_accuracy'])
    best_top1_accuracy = float('-inf')

# Check if results_df is a DataFrame
if not isinstance(results_df, pd.DataFrame):
    raise TypeError("results_df is not a DataFrame")

# Grid search
for epochs in epochs_grid:
    for imgsz in imgsz_grid:
        for lr0 in lr0_grid:
            for freeze in freeze_grid:
                for momentum in momentum_grid:
                    for label_smoothing in label_smoothing_grid:
                        for dropout in dropout_grid:
                            # Skip this combination if it's already been done
                            if any(row.equals(pd.Series({'epochs': epochs, 'imgsz': imgsz, 'lr0': lr0, 'freeze': freeze, 'momentum': momentum, 'label_smoothing': label_smoothing, 'dropout': dropout})) for _, row in results_df[['epochs', 'imgsz', 'lr0', 'freeze', 'momentum', 'label_smoothing', 'dropout']].iterrows()):
                                print(f'Skipping epochs={epochs}, imgsz={imgsz}, lr0={lr0}, freeze={freeze}, momentum={momentum}, label_smoothing={label_smoothing}, dropout={dropout}...')
                                continue
                            
                            print(f'Training with epochs={epochs}, imgsz={imgsz}, lr0={lr0}, freeze={freeze}, momentum={momentum}, label_smoothing={label_smoothing}, dropout={dropout}...')
                            
                            # Train the model
                            results = model.train(data='/Users/tobiasvontobel/Polybox/Master/Semester_4/Semester_Project/captcha_project/data', 
                                                  epochs=epochs, 
                                                  imgsz=imgsz, 
                                                  lr0=lr0, 
                                                  freeze=freeze, 
                                                  momentum=momentum, 
                                                  label_smoothing=label_smoothing, 
                                                  dropout=dropout, 
                                                  project=save_dir)
                            
                            # Check if this is the best model so far
                            top1_accuracy = results.top1
                            print(f'Top-1 accuracy: {top1_accuracy}')
                            if top1_accuracy > best_top1_accuracy:
                                best_top1_accuracy = top1_accuracy
                            
                            # Add the results to the dataframe and save it
                            new_row = pd.DataFrame([{'epochs': epochs, 'imgsz': imgsz, 'lr0': lr0, 'freeze': freeze, 'momentum': momentum, 'label_smoothing': label_smoothing, 'dropout': dropout, 'top1_accuracy': top1_accuracy}])
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
                            results_df.to_csv('grid_search_results.csv', index=False)

# Print the best hyperparameters
best_hyperparams = results_df.loc[results_df['top1_accuracy'].idxmax(), ['epochs', 'imgsz', 'lr0', 'freeze', 'momentum', 'label_smoothing', 'dropout']]
print(f'Best hyperparameters: {best_hyperparams}') 