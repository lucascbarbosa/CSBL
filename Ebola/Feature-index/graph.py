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
  nodes = []
  edges = []
  for outcome in outcomes.columns.values:
    for time_point in time_points:
      genes = []
      scores = []
      groups = []
      sources = []
      targets = []
      corrs = []
      group = 1
      node = pd.DataFrame()
      edge = pd.DataFrame()
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
          y = y.query(f"{outcome} == 'AE' | {outcome} == 'P'"  )[outcome] #AUTOMATIZAR OUTCOMES +
          y = y.loc[np.intersect1d(X.index.values,y.index.values)]
          X = X.loc[np.intersect1d(X.index.values,y.index.values)]
          X = X.corr()
          for j in range(len(X)):
            for i in range(len(X)):
              if i != j:
                if X.iloc[i,j] > tol:
                  sources.append(X.columns[j])
                  targets.append(X.index[i])
                  corrs.append(abs(X.iloc[i,j])) #MUDAR QUANDO DESCOBRIR COMO SETAR A COR DA ARESTA
            
      genes = np.array(genes).ravel()
      scores = np.array(scores).ravel()
      groups = np.array(groups).ravel()
      sources = np.array(sources).ravel()
      targets = np.array(targets).ravel()
      corrs = np.array(corrs).ravel()
      node['nodesize'] = scores 
      node['group'] = groups
      node['name'] = genes
      node['nodesize'] = node['nodesize']*100/node['nodesize'].max()
      node['nodesize'] = node['nodesize'].astype(int)
      edge['source'] = sources
      edge['target'] = targets
      edge['value'] = corrs
      nodes.append(node)
      edges.append(edge)
      

  return nodes,edges

  
def draw_graph(nodes,edges):

  for nodes_df,edges_df in list(zip(nodes,edges)):
    G = nx.Graph()

    for index, row in nodes_df.iterrows():
      G.add_node(row['name'], group=row['group'], nodesize=5*row['nodesize'])
        
    for index, row in edges_df.iterrows():
      G.add_weighted_edges_from([(row['source'], row['target'], 10*row['value'])])
        
    color_map = {1:'f09494', 2:'eebcbc', 3:'72bbd0', 4:'91f0a1'} 
    plt.figure(figsize=(30,30))
    options = {
      'edge_color': '#FFDEA2',
      'width': 4,
      'with_labels': True,
      'font_weight': 'regular',
      'font_size': 20
    }

    sizes = []

    for node in G:
        try: 
            if G.nodes[node]!={}:
                sizes.append(G.nodes[node]['nodesize']*5)
        except: 
            pass

    nx.draw(G, node_size=sizes, pos=nx.spring_layout(G, k=2), **options)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#555555") 
    # plt.savefig('C:/Users/lucas/Documents/GitHub/PatentsAnalytics/Visualization/grafos.png')
    plt.show()

nodes,edges = get_nodes_edges(time_points,features_score_path,outcomes)
draw_graph(nodes,edges)