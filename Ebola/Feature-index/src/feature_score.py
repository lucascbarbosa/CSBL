import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer 

#Extract dataframe
df = pd.read_csv('C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Top_Features/RNASeq_AdverseEvent_TOPF100.csv').T
df.columns = df.iloc[0]
df.drop(['Probes'],axis=0,inplace=True)
X = df.drop(['Class'],axis=1)
genes = X.columns
y = df['Class']
lb = LabelBinarizer()
y = lb.fit_transform(y).ravel()

#Get scores
trees = 100
model = RandomForestClassifier(n_estimators=trees)
model.fit(X, y)
importance = model.feature_importances_
scores = pd.DataFrame()
scores['Score'] = importance
scores.index = genes
scores.sort_values(by=['Score'])
scores['Rank'] = range(1,len(scores)+1)
scores.to_csv(f'C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Feature_Scores/RNASeq_AdverseEvent_TOPF100_SCORES.csv')