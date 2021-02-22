from top_features import top_features
from pathlib import Path
import sys

import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedShuffleSplit

from imblearn.over_sampling import SMOTE

import mlflow

def get_hiperparams(X_train,X_test,y_train,y_test,kfold,param_grid, model):
  gs = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=kfold)
  best_gs = gs.fit(X_train,y_train)
  n = len(param_grid)
  scores = best_gs.cv_results_['mean_test_score']
  scores = scores.ravel()
  return best_gs.best_params_


input_file_X = sys.argv[1]
input_file_y = sys.argv[2]
filtered_input_file = sys.argv[3]
coly = sys.argv[4]
classifier = sys.argv[5]
top = int(sys.argv[6])
test_size = float(sys.argv[7])
n_splits = int(sys.argv[8])
file_format = input_file_X.split('/')[-1][-3:]

X = top_features(input_file_X,coly,file_format,top,input_file_y=input_file_y,filtered_input_file=filtered_input_file,save=True)
y = X['Class']
X.drop(['Class'],axis=1,inplace=True)
lb = LabelBinarizer()
y = pd.DataFrame(lb.fit_transform(y).ravel())
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

kfold = KFold(n_splits,shuffle=True,random_state=42)

#SVC
param_grid_svc = {'C':[1, 5, 10, 50, 100,],
                  'gamma':['scale','auto']
                }

#Decision Trees
param_grid_tree = {'max_depth':[1,2,3,4,5],
                  'criterion':['gini','entropy']
                  }

#KNN
param_grid_knn = {'n_neighbors':[1,2,3,4,5],
                  'algorithm':['auto','ball_tree','kd_tree','brute']
                }

#Random Forest
param_grid_rf = {'n_estimators':[150,158,166,174,182]}

#SGD
param_grid_sgd = {'loss':['hinge', 'perceptron'],
                'penalty':['l2', 'l1', 'elasticnet'],
                'alpha':[1e-5,1e-4,1e-3,1e-2,1e-1]
                }


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)

if classifier == 'SVM':
  hp_svc = get_hiperparams(X_train,X_test,y_train,y_test,kfold,param_grid_svc,SVC())
  c_opt,gamma_opt = hp_svc.values()
  svc = make_pipeline(StandardScaler(),
                      SVC(C=c_opt,
                          gamma=gamma_opt,
                          kernel='rbf'
                        )
                    )
  svc.fit(X_train,y_train)
  y_pred = svc.predict(X_test)
  cm = list(confusion_matrix(y_test,y_pred).astype(str).ravel())

# if classifier == 'KNN':

# if classifier == 'SGD':

# if classifier == 'DecisionTree':

# if classifier == 'RandomForest':
cm = '-'.join(cm)
cm = cm.encode('ascii')
print(cm,end='')