from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def impute(input_file_X,neighbors,file_format):
  if file_format == 'txt':
    X = pd.read_csv(input_file_X, sep='\t', header=0, index_col=0).astype(np.float64).round(9)
  if file_format == 'csv':
    X = pd.read_csv(input_file_X, header=0, index_col=0).astype(np.float64).round(9)
  imputer = KNNImputer(n_neighbors=neighbors)
  cols = X.columns
  idx = X.index
  X = imputer.fit_transform(X)
  X = pd.DataFrame(X,index=idx,columns=cols).round(9)
  if file_format == 'txt':
    X.to_csv(input_file_X, sep='\t', header=True, index=True)
  if file_format == 'csv':
    X.to_csv(input_file_X, header=True, index=True)