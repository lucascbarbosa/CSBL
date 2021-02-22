import pandas as pd

def get_best_window(df,size):
  best_score = 0.0
  best_window = [df.index.values[0],df.index.values[1]]
  for i in range(len(df)-size):
    data = df.iloc[i:i+size,:]
    tpr = data['TP'].values/(data['TP'].values+data['FN'].values)
    tpr = tpr.sum()/size
    # fpr = data['FP'].values/(data['TN'].values+data['FP'].values)
    # fpr = fpr.sum()/size
    # score = float(tpr/fpr)
    score = tpr
    if score > best_score:
      best_score = score
      best_window = [data.index.values[0],data.index.values[-1]]

  return best_window
