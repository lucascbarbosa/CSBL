import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer 

def txt2arff(filepath_X,filepath_y):
  X = pd.read_csv(filepath_X, sep=' ', header=0, index_col= 0).astype(float)
  X = X.round(6)
  y = pd.read_csv(filepath_y, sep=' ', header=0, index_col= 0)
  lb = LabelBinarizer()
  y = lb.fit_transform(y).ravel().astype(int)
  df = X
  df['Class'] = y

  arff.dump('teste.arff',
            df.values, 
            names=df.columns)