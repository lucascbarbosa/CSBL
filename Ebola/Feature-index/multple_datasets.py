import pandas as pd
from src.input2arff import input2arff
from src.top_features import top_features
from src.features_score import features_score
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

top = int(sys.argv[1])
trees = int(sys.argv[2])
tol = float(sys.argv[3])
neighbors = int(sys.argv[4])
try:
  outcomes = pd.read_csv('Data/Input/Outcomes.txt', sep='\t', header=0,index_col=0).columns.values
except:
  outcomes = pd.read_csv('Data/Input/Outcomes.csv',header=0,index_col=0).columns.values
  
for path in os.listdir('Data/Input/'):
  for outcome in outcomes:
    if path[:-4] == 'Outcomes':
      pass
    else:
      pathX = path
      file_format = pathX[-3:]
      input_file_X = Path('Data/Input/'+pathX).absolute()
      path_input = input_file_X.parents[0]
      input_file_y = Path(path_input,'Outcomes.'+file_format)
      filtered_input_file = Path(input_file_X.parents[1],'Filtered_Input/'+pathX)
      top_features_file = Path(input_file_X.parents[1],'Top_Features/'+pathX[:-4]+'_'+outcome+'_TOP%d.'%(top)+file_format)
      features_score_file = Path(input_file_X.parents[1],'Features_Score/'+pathX[:-4]+'_'+outcome+'_TOP%d_SCORES.'%(top)+file_format)
      print('\n',path,outcome,'\n')
      # input2arff(input_file_X,input_file_y,filtered_input_file,outcome,file_format)
      # top_features(input_file_X,input_file_y,filtered_input_file,top_features_file,outcome,file_format,top)
      # features_score(top_features_file,features_score_file,file_format,trees)
      command_filter = 'python filter.py %s %d %d'%(path, tol, neighbors)
      os.system(command_filter)
      command_feature = 'python single_dataset.py %s %s %d %d'%(path,outcome,top,trees)
      os.system(command_feature)