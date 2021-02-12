import pandas as pd
import numpy as np
import os

pathX = 'C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Input'
pathY = 'C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Input/Outcomes.txt'

Y = pd.read_csv(pathY, sep='\t', header=0, index_col= 0)
Y.to_csv(pathY[:-4]+'.csv', header=True, index=True)

for path in os.listdir(pathX)[2:]:
  
  X = pd.read_csv(pathX+'/'+path, sep='\t', header=0, index_col= 0).astype(float).round(6)
  X.to_csv(pathX+'/'+path[:-4]+'.csv', header=True, index=True)