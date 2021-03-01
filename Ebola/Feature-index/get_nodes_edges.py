from src.get_scores import get_scores
import os
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from scipy.stats import spearmanr
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

tol = float(sys.argv[1])

time_points = [f'D{i}vD0'for i in [1,3,7,14,28]]
features_score_path = 'Data/Features_Score/'
try:
  outcomes = pd.read_csv('Data/Input/Outcomes.txt', sep='\t', header=0,index_col=0)
except:
  outcomes = pd.read_csv('Data/Input/Outcomes.csv',header=0,index_col=0)


def get_nodes_edges(time_points,features_score_path,outcomes):
  for outcome in outcomes.columns.values:
    for time_point in time_points:
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
      corrX = spearmanr(X)[0] #spearman correlation

      for j in range(len(corrX)):
        for i in range(len(corrX)):
          if i != j:
            if corrX[i,j] > tol:
              edges_group.append(1)
              sources.append(X.columns[i])
              targets.append(X.columns[j])
              corrs.append(abs(corrX[i,j]))
            elif corrX[i,j] < -tol:
              edges_group.append(0)
              sources.append(X.columns[i])
              targets.append(X.columns[j])
              corrs.append(abs(corrX[i,j]))

      relevant_genes = []
      relevant_groups = []
      relevant_scores = []

      for gene,group,score in list(zip(genes,groups,scores)):
        if gene in sources or gene in targets:
          relevant_genes.append(gene)
          relevant_groups.append(group)
          relevant_scores.append(score)
        else:
          pass

      relevant_genes = np.array(relevant_genes).ravel()
      relevant_groups = np.array(relevant_groups).ravel()
      relevant_scores = np.array(relevant_scores).ravel()
      sources = np.array(sources).ravel()
      targets = np.array(targets).ravel()
      corrs = np.array(corrs).ravel()
      node['label'] = relevant_genes
      node['group'] = relevant_groups
      node['nodesize'] = relevant_scores 
      node['nodesize'] = node['nodesize'].astype(int)
      edge['source'] = sources
      edge['target'] = targets
      edge['Weight'] = corrs
      edge['group'] = edges_group
      node.index = relevant_genes
      node.index.name = 'id'
      edge.index.name = 'id'
      node.to_csv(f'Data/Nodes_Edges/{time_point}_{outcome}_nodes.csv')
      edge.to_csv(f'Data/Nodes_Edges/{time_point}_{outcome}_edges.csv')


get_nodes_edges(time_points,features_score_path,outcomes)