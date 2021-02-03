from src.txt2arff import txt2arff
from src.top_features import top_features
from src.feature_scores import feature_scores
import sys

input_file_X = sys.argv[1]
TOP = sys.argv[2]
trees = sys.argv[3]
# input_file_X = 'C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Input/RNASeq_AdverseEvent.txt'
input_file_y = input_file_X[:-4]+'_Class.txt'
top_features_file = '/'.join(input_file_X.split('/')[:-2])+'/Top_Features/'+input_file_X.split('/')[-1][:-4]+f'_TOP{TOP}.txt'
feature_scores_file = '/'.join(input_file_X.split('/')[:-2])+'/Features_Scores/'+input_file_X.split('/')[-1][:-4]+f'_TOP{TOP}_SCORES.txt'
print(input_file_X,TOP,trees)