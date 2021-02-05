import csv2arff
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def input2arff(input_file_X,input_file_y,filtered_input_file,coly,file_format):
  if file_format == 'txt':
    X = pd.read_csv(input_file_X, sep='\t', header=0, index_col= 0).T.astype(float).round(6)
    y = pd.read_csv(input_file_y, sep='\t', header=0, index_col= 0)
  elif file_format == 'csv':
    X = pd.read_csv(input_file_X, header=0, index_col= 0).T.astype(float).round(6)
    y = pd.read_csv(input_file_y, header=0, index_col= 0)
  else:
    raise('Erro: Somente arquivos txt ou csv.')
    return None
  y = y[coly]
  # X = X.loc[y.index]
  X = X.loc[np.intersect1d(X.index.values,y.index.values)]
  y = y.loc[np.intersect1d(X.index.values,y.index.values)]
  df = X
  df.index.name = 'Probes'
  df['Class'] = y.values
  path_csv = str(filtered_input_file)[:-4]+'.csv'
  df.to_csv(path_csv, header=True, index=True)
  path_arff = str(filtered_input_file)[:-4]+'.arff'
  command = 'csv2arff %s %s'%(path_csv,path_arff)
  os.system(command)