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
classifier = sys.argv[3]
top_max = int(sys.argv[4])
test_size = float(sys.argv[5])
n_splits = int(sys.argv[6])

input_file_X = Path('Data/Input/'+pathX).absolute()
path_input = input_file_X.parents[0]
filtered_input_file = Path(input_file_X.parents[1],'Filtered_Input/'+pathX[:-4]+'_'+coly+'.'+file_format)
input_file_y = Path(path_input,'Outcomes.'+file_format)

tops = [int(i*0.1*top_max) for i in range(1,11)]
cms = []
cm_df = pd.DataFrame(columns='TN FP FN TP'.split())
cols = cm_df.columns.values

for top in tops:
  command_classify = 'cd src && python classify.py %s %s %s %s %s %d %f %d && cd..'%(input_file_X,input_file_y,filtered_input_file,coly,classifier,top,test_size,n_splits)
  # os.system(command_classify)
  sub = Popen(command_classify, shell=True, stdout=PIPE, stderr=STDOUT)
  cm = sub.stdout.read()[-9:-1].decode('ascii')
  cm = [int(i) for i in cm.split('-')]
  cms.append(cm)

cms = np.array(cms)
for i in range(4):
  cm_df[cols[i]] = cms[:,i]

cm_df.index = tops
print(cm_df)