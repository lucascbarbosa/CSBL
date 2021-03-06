from src.remove_NA import remove_NA
from src.impute import impute
from pathlib import Path
import sys

pathX = sys.argv[1]
file_format = pathX[-3:]
tol = float(sys.argv[2])
neighbors = int(sys.argv[3])
input_file_X = Path('Data/Input/'+pathX).absolute()

remove_NA(input_file_X,tol,file_format)
impute(input_file_X,neighbors,file_format)