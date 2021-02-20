from src.input2arff import input2arff
from src.top_features import top_features
from src.features_score import features_score
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

pathX = sys.argv[1]
file_format = pathX[-3:]
coly = sys.argv[2]
top = int(sys.argv[3])
trees = int(sys.argv[4])

input_file_X = Path('Data/Input/'+pathX).absolute()
path_input = input_file_X.parents[0]
input_file_y = Path(path_input,'Outcomes.'+file_format)
filtered_input_file = Path(input_file_X.parents[1],'Filtered_Input/'+pathX[:-4]+'_'+coly+'.'+file_format)
top_features_file = Path(input_file_X.parents[1],'Top_Features/'+pathX[:-4]+'_'+coly+'_TOP%d.'%(top)+file_format)
features_score_file = Path(input_file_X.parents[1],'Features_Score/'+pathX[:-4]+'_'+coly+'_TOP%d_SCORES.'%(top)+file_format)

input2arff(input_file_X,input_file_y,filtered_input_file,coly,file_format)
X,y = top_features(input_file_X,coly,file_format,top,input_file_y,filtered_input_file,top_features_file)
features_score(top_features_file,features_score_file,file_format,trees)