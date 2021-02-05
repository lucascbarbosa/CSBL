import pandas as pd
import numpy as np

pathX = 'C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Input/RNAseq_logCPM_D7vD0.txt'
pathY = 'C:/Users/lucas/Documents/Github/CSBL/Ebola/Feature-index/Data/Input/Outcomes.txt'

X = pd.read_csv(pathX, sep='\t', header=0, index_col= 0).astype(float).round(6)
Y = pd.read_csv(pathY, sep='\t', header=0, index_col= 0)

X.to_csv(pathX[:-4]+'.csv', header=True, index=True)
Y.to_csv(pathY[:-4]+'.csv', header=True, index=True)
