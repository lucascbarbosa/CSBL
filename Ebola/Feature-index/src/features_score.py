import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer 


def features_score(top_features_file,features_score_file,trees):
  #Extract dataframe
  df = pd.read_csv(top_features_file,sep=' ', header=0, index_col=0)
  X = df.drop(['Class'],axis=1)
  genes = X.columns
  y = df['Class']

  #Get scores
  model = RandomForestClassifier(n_estimators=trees)
  model.fit(X, y)
  importance = model.feature_importances_
  scores = pd.DataFrame()
  scores['Score'] = importance
  scores.index = genes
  scores.sort_values(by=['Score'],inplace=True,ascending=False)
  scores['Rank'] = range(1,len(scores)+1)
  scores.to_csv(features_score_file, sep=' ', header=True, index=True)