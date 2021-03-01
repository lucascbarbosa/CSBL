from src.input2arff import input2arff
from src.top_features import top_features
from src.features_score import features_score
import sys
import os
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
import warnings
warnings.filterwarnings("ignore")

pathX = sys.argv[1]
file_format = pathX[-3:]
coly = sys.argv[2]
top_max = int(sys.argv[3])
trees = int(sys.argv[4])
test_size = float(sys.argv[5])
metric = sys.argv[6]

input_file_X = Path('Data/Input/'+pathX).absolute()
path_input = input_file_X.parents[0]
input_file_y = Path(path_input,'Outcomes.'+file_format)

filtered_input_file = Path(input_file_X.parents[1],'Filtered_Input/'+pathX[:-4]+'_'+coly+'.'+file_format)
input2arff(input_file_X,input_file_y,filtered_input_file,coly,file_format)

command_top = 'python get_best_top.py %s %s %d %d %f  %s'%(pathX,coly,trees,top_max,test_size,metric)
sub = Popen(command_top, shell=True, stdout=PIPE, stderr=STDOUT)
top = int(sub.stdout.read())

top_features_file = Path(input_file_X.parents[1],'Top_Features/'+pathX[:-4]+'_'+coly+'_TOP%d.'%(top)+file_format)
top_features(input_file_X,coly,file_format,top,input_file_y,filtered_input_file,top_features_file,True)
features_score_file = Path(input_file_X.parents[1],'Features_Score/'+pathX[:-4]+'_'+coly+'_TOP')
features_score(top_features_file,features_score_file,file_format,trees)