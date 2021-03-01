import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

def features_score(top_features_file,features_score_file,file_format,trees):
  #Extract dataframe
  if file_format == 'txt':
    df = pd.read_csv(top_features_file,sep='\t', header=0, index_col=0)
    X = df.drop(['Class'],axis=1).astype(np.float64).round(9)
    genes = X.columns
    y = df['Class']
  if file_format == 'csv':
    df = pd.read_csv(top_features_file, header=0, index_col=0)
    X = df.drop(['Class'],axis=1).astype(np.float64).round(9)
    genes = X.columns
    y = df['Class']
    
  #Get scores
  model = RandomForestClassifier(n_estimators=trees,random_state=42)
  model.fit(X, y)
  importance = model.feature_importances_.reshape(-1,1)
  scores = pd.DataFrame()
  scaler  = MinMaxScaler()
  scores['Score'] = scaler.fit_transform(importance).ravel()
  scores['Score'] = scores['Score']*len(scores['Score'])
  scores.index = genes
  scores.sort_values(by=['Score'],inplace=True,ascending=False)
  scores['Rank'] = range(1,len(scores)+1)
  top_ = len(scores[scores['Score']!=0.0])
  features_score_file  = str(features_score_file) + f'{top_}.' + file_format

  if file_format == 'txt':
    scores[scores['Score']!=0.0].to_csv(features_score_file, sep='\t', header=True, index=True)
  if file_format == 'csv':
    scores[scores['Score']!=0.0].to_csv(features_score_file, header=True, index=True)
