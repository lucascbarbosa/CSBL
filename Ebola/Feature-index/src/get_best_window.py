import pandas as pd
from sklearn.metrics import auc

def get_best_window(df,size,mode):
  best_window = [df.index.values[0],df.index.values[1]] 

  if mode == 'tpr':
    best_score = 0.0  
    for i in range(len(df)-size):
      data = df.iloc[i:i+size,:-1]
      tpr = data['TP'].values/(data['TP'].values+data['FN'].values)
      tpr = tpr.sum()/size
      # fpr = data['FP'].values/(data['TN'].values+data['FP'].values)
      # score = auc(tpr.fpr)
      if tpr > best_score:
        best_score = tpr
        best_window = [data.index.values[0],data.index.values[-1]]

  elif mode == 'fpr':
    best_score = 1.0
    for i in range(len(df)-size):
      data = df.iloc[i:i+size,:-1]
      fpr = data['FP'].values/(data['TN'].values+data['FP'].values)
      tpr = tpr.sum()/size
      if fpr < best_score:
        best_score = fpr
        best_window = [data.index.values[0],data.index.values[-1]]
  
  elif mode == 'auc':
    best_score = 0.0
    for i in range(len(df)-size):
      auc = df.iloc[i:i+size,-1]
      auc = auc.sum()/size
      if auc < best_score:
        best_score = auc
        best_window = [data.index.values[0],data.index.values[-1]]

  return best_window
