"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")
DATA_PATH = "data/"

# 1. TimeGAN model
from timegan_v2 import timegan
# 2. Data loading
from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
    inverse_transform_data,
    save_scaler,
    save_data,
)
# 3. Metrics
# from metrics.discriminative_metrics import discriminative_score_metrics
# from metrics.predictive_metrics import predictive_score_metrics
# from metrics.visualization_metrics import visualization


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: name of dataset
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data Loading
  # read data
  data = load_data(data_dir=DATA_PATH, dataset=args.data_name)

  # split data into train/valid splits
  train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)

  # scale data
  scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)
  
  # Adding datasets back together for now since this pipeline prefers a single dataset
  ori_data = np.concatenate([scaled_train_data, scaled_valid_data])
  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
      
  generated_data = timegan(ori_data, parameters)   
  print('Finish Synthetic Data Generation')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # # 1. Discriminative Score
  # discriminative_score = list()
  # for _ in range(args.metric_iteration):
  #   temp_disc = discriminative_score_metrics(ori_data, generated_data)
  #   discriminative_score.append(temp_disc)
      
  # metric_results['discriminative'] = np.mean(discriminative_score)
      
  # # 2. Predictive score
  # predictive_score = list()
  # for tt in range(args.metric_iteration):
  #   temp_pred = predictive_score_metrics(ori_data, generated_data)
  #   predictive_score.append(temp_pred)   
      
  # metric_results['predictive'] = np.mean(predictive_score)     
          
  # # 3. Visualization (PCA and tSNE)
  # visualization(ori_data, generated_data, 'pca')
  # visualization(ori_data, generated_data, 'tsne')
  
  ## Print discriminative and predictive scores
  print(metric_results)

  return ori_data, generated_data, metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=30,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  
  # args = parser.parse_args() 
  args = parser.parse_args(args=[
      '--data_name', 'jerkEvents_25',
      '--seq_len', '24',
      '--module', 'gru',
      '--hidden_dim', '24',
      '--num_layer', '3',
      '--iteration', '20',
      '--batch_size', '128',
      '--metric_iteration', '10'
  ])
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)