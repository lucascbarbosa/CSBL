import csv2arff
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer 

def txt2arff(input_file_X,input_file_y):
  X = pd.read_csv(input_file_X, sep=' ', header=0, index_col= 0).astype(float)
  X = X.round(6)
  y = pd.read_csv(input_file_y, sep=' ', header=0, index_col= 0)
  # lb = LabelBinarizer()
  # y = lb.fit_transform(y).ravel().astype(int)
  df = X
  df['Class'] = y.values
  df.to_csv(input_file_X[:-4]+'.csv',header=True, index=True)
  command = 'csv2arff %s.csv %s.arff'%(input_file_X[:-4],input_file_X[:-4])
  os.system(command)
  

TOP = 20
input_file_X = 'C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Input/RNASeq_AdverseEvent.txt'
input_file_y = input_file_X[:-4]+'_Class.txt'