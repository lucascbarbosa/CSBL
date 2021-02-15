from src.get_scores import get_scores
import os
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from yellowbrick.target import FeatureCorrelation
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

tol = float(sys.argv[1])

time_points = [f'D{i}vD0'for i in [1,3,7,14,28,180,365]]
features_score_path = 'Data/Features_Score/'
try:
  outcomes = pd.read_csv('Data/Input/Outcomes.txt', sep='\t', header=0,index_col=0)
except:
  outcomes = pd.read_csv('Data/Input/Outcomes.csv',header=0,index_col=0)

def get_nodes_edges(time_points,features_score_path,outcomes):
  for outcome in outcomes.columns.values:
    for time_point in time_points:
      # print(outcome,time_point,'\n')
      genes = []
      scores = []
      groups = []
      sources = []
      targets = []
      corrs = []
      edges_group = []
      group = 1
      node = pd.DataFrame()
      edge = pd.DataFrame()
      Xs = []
      for features_score_file in os.listdir(features_score_path):
        if time_point in features_score_file and outcome in features_score_file:
          file_format = features_score_file[-3:] 
          _genes,_scores = get_scores(features_score_path,features_score_file,file_format)
          genes += _genes
          scores += _scores
          groups += [group for i in range(len(_genes))]
          group += 1

          if file_format == 'txt':
            input_file = '_'.join(features_score_file.split('_')[:-4])+'.'+file_format
            X = pd.read_csv('Data/Input/'+input_file, sep='\t',header=0, index_col=0).T
          elif file_format == 'csv':
            X = pd.read_csv('Data/Input/'+input_file, header=0, index_col=0).T
          X = X[_genes]
          y = outcomes
          y = y.query(f"{outcome} == 'AE' | {outcome} == 'P'")[outcome] #AUTOMATIZAR OUTCOMES +
          y = y.loc[np.intersect1d(X.index.values,y.index.values)]
          X = X.loc[np.intersect1d(X.index.values,y.index.values)]
          Xs.append(X)

      X = pd.concat(Xs,axis=1)
      X = X.corr()
      for j in range(len(X)):
        for i in range(len(X)):
          if i != j:
            if X.iloc[i,j] > tol:
              edges_group.append(1)
              sources.append(X.columns[j])
              targets.append(X.index[i])
              corrs.append(abs(X.iloc[i,j]))
            elif X.iloc[i,j] < -tol:
              edges_group.append(0)
              sources.append(X.columns[j])
              targets.append(X.index[i])
              corrs.append(abs(X.iloc[i,j]))

      genes = np.array(genes).ravel()
      scores = np.array(scores).ravel()
      groups = np.array(groups).ravel()
      sources = np.array(sources).ravel()
      targets = np.array(targets).ravel()
      corrs = np.array(corrs).ravel()

      node['label'] = genes
      node['group'] = groups
      node['nodesize'] = scores 
      node['nodesize'] = node['nodesize']*100/node['nodesize'].max()
      node['nodesize'] = node['nodesize'].astype(int)
      edge['source'] = sources
      edge['target'] = targets
      edge['Weight'] = corrs
      edge['group'] = edges_group
      node.index = genes
      node.index.name = 'id'
      edge.index.name = 'id'
      node.to_csv(f'Data/Nodes_Edges/nodes_{time_point}_{outcome}.csv')
      edge.to_csv(f'Data/Nodes_Edges/edges_{time_point}_{outcome}.csv')

get_nodes_edges(time_points,features_score_path,outcomes)