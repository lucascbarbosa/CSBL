import pandas as pd
import numpy as np

def remove_NA(input_file_X,tol,file_format):
  if file_format == 'txt':
    X = pd.read_csv(input_file_X, sep='\t', header=0, index_col=0).astype(np.float64).round(9)
  if file_format == 'csv':
    X = pd.read_csv(input_file_X, header=0, index_col=0).astype(np.float64).round(9)
  
  probes = len(X.columns)
  for gene in X.index:
    nans = np.where(X.loc[gene].isna().values==True)[0].shape[0]
    if float(nans/probes) > tol:
      X.drop([gene],axis=0,inplace=True)
  genes = len(X)
  for probe in X.columns:
    nans = np.where(X[probe].isna().values==True)[0].shape[0]
    if float(nans/genes) > tol:
      X.drop([probe],axis=1,inplace=True)
  if file_format == 'txt':
    X.to_csv(input_file_X, sep='\t', na_rep='NA', header=True, index=True)
  if file_format == 'csv':
    X.to_csv(input_file_X, na_rep='NA', header=True, index=True)
  