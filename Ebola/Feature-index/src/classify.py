from top_features import top_features
from pathlib import Path
import sys

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

input_file_X = sys.argv[1]
input_file_y = sys.argv[2]
filtered_input_file = sys.argv[3]
coly = sys.argv[4]
trees = int(sys.argv[5])
top = int(sys.argv[6])
test_size = float(sys.argv[7])
file_format = input_file_X.split('/')[-1][-3:]

X = top_features(input_file_X,coly,file_format,top,input_file_y=input_file_y,filtered_input_file=filtered_input_file,save=True)
y = X['Class']
X.drop(['Class'],axis=1,inplace=True)
lb = LabelBinarizer()
y = pd.DataFrame(lb.fit_transform(y).ravel())
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)

rf = make_pipeline(StandardScaler(),
                    RandomForestClassifier(n_estimators=trees,random_state=42)
                  )
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
cm = list(confusion_matrix(y_test,y_pred).astype(str).ravel())
cm = '-'.join(cm)
print(cm,end='')