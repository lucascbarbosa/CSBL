import pandas as pd
from src.input2arff import input2arff
from src.top_features import top_features
from src.features_score import features_score
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

top_max = int(sys.argv[1])
trees = int(sys.argv[2])
tol = float(sys.argv[3])
neighbors = int(sys.argv[4])
test_size = float(sys.argv[5])
metric = sys.argv[6]

try:
  outcomes = pd.read_csv('Data/Input/Outcomes.txt', sep='\t', header=0,index_col=0).columns.values
except:
  outcomes = pd.read_csv('Data/Input/Outcomes.csv',header=0,index_col=0).columns.values
  
for path in os.listdir('Data/Input/'):
  for outcome in outcomes:
    if path[:-4] == 'Outcomes':
      pass
    else:
      print(path,outcome)
      command_filter = 'python filter.py %s %f %d'%(path, tol, neighbors)
      os.system(command_filter)
      command_feature = 'python single_dataset.py %s %s %d %d %f %s'%(path,outcome,top_max,trees,test_size,metric)
      os.system(command_feature)