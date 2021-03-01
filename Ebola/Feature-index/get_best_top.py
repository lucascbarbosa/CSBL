from src.get_best_window import get_best_window
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
import warnings
warnings.filterwarnings("ignore")

pathX = sys.argv[1]
coly = sys.argv[2]
file_format = pathX[-3:]
trees = int(sys.argv[3])
top_max = int(sys.argv[4])
test_size = float(sys.argv[5])
metric = sys.argv[6]

input_file_X = Path('Data/Input/'+pathX).absolute()
path_input = input_file_X.parents[0]
filtered_input_file = Path(input_file_X.parents[1],'Filtered_Input/'+pathX[:-4]+'_'+coly+'.'+file_format)
input_file_y = Path(path_input,'Outcomes.'+file_format)


def iterate_tops(tops,input_file_X,input_file_y,filtered_input_file,coly,test_size):
  cms = []
  metrics_df = pd.DataFrame(columns='TN FP FN TP AUC'.split())
  cols = metrics_df.columns.values
  aucs = []
  for top in tops:
    command_classify = 'cd src && python classify.py %s %s %s %s %d %d %f && cd..'%(input_file_X,input_file_y,filtered_input_file,coly,trees,top,test_size)
    sub = Popen(command_classify, shell=True, stdout=PIPE, stderr=STDOUT)
    cm_auc = sub.stdout.read().decode('ascii').split('\n')[-1]
    cm = cm_auc.split('-')[:-1]
    cm = [int(i) for i in cm]
    auc = float(cm_auc.split('-')[-1])
    aucs.append(auc)
    cms.append(cm)

  cms = np.array(cms)
  for i in range(4):
    metrics_df[cols[i]] = cms[:,i]
  
  metrics_df[cols[4]] = aucs

  metrics_df.index = tops
  l_top, u_top = get_best_window(metrics_df,2,metric)
  return l_top, u_top

if 0.01*float(top_max) > 5:
  intervals = [0.1,0.01]
else:
  intervals = [0.1,float(5/top_max)]
  
l_top = int(intervals[0]*top_max)
u_top = top_max

for i in range(len(intervals)):
  tops = list(range(l_top,u_top+1,int(intervals[i]*top_max)))
  l_top, u_top = iterate_tops(tops,input_file_X,input_file_y,filtered_input_file,coly,test_size)
  
print(l_top, end='')