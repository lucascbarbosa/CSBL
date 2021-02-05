from src.txt2arff import txt2arff
from src.top_features import top_features
from src.features_score import features_score
import sys

input_file_X = sys.argv[1]
top = int(sys.argv[2])
trees = int(sys.argv[3])
# input_file_X = 'C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Input/RNASeq_AdverseEvent.txt'
input_file_y = input_file_X[:-4]+'_Class.txt'
top_features_file = '/'.join(input_file_X.split('/')[:-2])+'/Top_Features/'+input_file_X.split('/')[-1][:-4]+'_TOP%d.txt'%(top)
features_score_file = '/'.join(input_file_X.split('/')[:-2])+'/Features_Score/'+input_file_X.split('/')[-1][:-4]+'_TOP%d_SCORES.txt'%(top)

txt2arff(input_file_X,input_file_y)
top_features(input_file_X,input_file_y,top_features_file,top)
features_score(top_features_file,features_score_file,trees)