import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

pathX = sys.argv[1]
coly = sys.argv[2]
file_format = pathX[-3:]
classifier = sys.argv[3]
top_max = int(sys.argv[4])
n_splits = int(sys.argv[5])

input_file_X = Path('Data/Input/'+pathX).absolute()
path_input = input_file_X.parents[0]
filtered_input_file = Path(input_file_X.parents[1],'Filtered_Input/'+pathX[:-4]+'_'+coly+'.'+file_format)
input_file_y = Path(path_input,'Outcomes.'+file_format)

tops = [int(i*0.1*top_max) for i in range(1,11)]

# input_file_X = sys.argv[1]
# coly = sys.argv[2]
# file_format = pathX[-3:]
# path_input = input_file_X.parents[0]
# filtered_input_file = Path(input_file_X.parents[1],'Filtered_Input/'+pathX[:-4]+'_'+coly+'.'+file_format)
# input_file_y = Path(path_input,'Outcomes.'+file_format)
# classifier = sys.argv[3] 
# top = int(sys.argv[4])
# n_splits= int(sys.argv[5])


for top in tops:
  print('\n'+str(top)+'\n')
  command_classify = 'cd src && python classify.py %s %s %s %s %s %d %d && cd..'%(input_file_X,input_file_y,filtered_input_file,coly,classifier,top,n_splits)
  os.system(command_classify)
