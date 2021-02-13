import pandas as pd
import os


def get_scores(features_score_path,features_score_file,file_format):
  total_scores = pd.DataFrame()
  genes = []

  if file_format == 'txt':
    df = pd.read_csv(features_score_path+features_score_file, sep='\t',header= 0, index_col=0)
  elif file_format == 'csv':
    df = pd.read_csv(features_score_path+features_score_file, header= 0, index_col=0)
  else:
    print(f'Erro. Arquivo {features_score_file} não possui formato aceito (csv ou txt).')
  
  return list(df.index),list(df['Score'])
