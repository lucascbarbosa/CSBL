import weka.core.jvm as jvm
import weka.core.converters as converters
import csv2arff as arff
import os
import csv
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from weka.core.converters import Loader, Saver
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer

# Argumentos de Entrada
Arquivo_Features = "data/RNAseq_logCPM_D1vD0.txt"
Arquivo_Class = "data/Outcomes.txt"
Name_Class = "AdverseEvent_Class"
TOP = 100

jvm.start()

comand_class = "cat %s | tr -s '[:blank:]' ',' > data/Outcomes01.txt" % (Arquivo_Class)
os.system(comand_class)
read_file = pd.read_csv (r"data/Outcomes01.txt")
read_file.to_csv (r"data/Outcomes01.csv", index = None)
os.system("rm data/Outcomes01.txt")

comand_features = "cat %s | tr -s '[:blank:]' ',' > data/Arq_Features01.txt" % (Arquivo_Features)
os.system(comand_features)
read_file = pd.read_csv (r"data/Arq_Features01.txt")
read_file.to_csv (r"data/Arq_Features01.csv", index = None)
os.system("rm data/Arq_Features01.txt")

arquivo_class = []
with open("data/Outcomes01.csv", "r") as arquivo_csv:
    leitor = csv.reader(arquivo_csv)
    for coluna in leitor:
        arquivo_class.append(coluna)

os.system("rm data/Outcomes01.csv")
class_selected = arquivo_class[0].index(Name_Class)
arquivo_class.sort()
count = 0

while count < len(arquivo_class):
	if arquivo_class[count][class_selected] == '':
		del(arquivo_class[count])
		count += 1
	else:
		count += 1

del(arquivo_class[-1])
arquivo_class_temp = []
count = 0
while count < len(arquivo_class):
	arquivo_class_temp.append(arquivo_class[count][class_selected])
	count += 1
arquivo_class_temp.insert(0, "Class")

arquivo_features = []
with open("data/Arq_Features01.csv", "r") as arquivo_csv:
    leitor = csv.reader(arquivo_csv)
    for coluna in leitor:
        arquivo_features.append(coluna)

os.system("rm data/Arq_Features01.csv")
arquivo_features = list(map(list, zip(*arquivo_features)))
arquivo_features_temp = arquivo_features[0]
arquivo_features.sort()

arquivo_temp = []

count = 0
while count < len(arquivo_class):
	count2 = 0
	while count2 < len(arquivo_features):
		if arquivo_class[count][0] == arquivo_features[count2][0]:
			arquivo_temp.append(arquivo_features[count2])
			count += 1
			break
		else:
			count2 +=1

arquivo_temp.insert(0, arquivo_features_temp)
arquivo_temp = list(map(list, zip(*arquivo_temp)))
arquivo_temp.append(arquivo_class_temp)

with open("data/ArquivoEntrada.csv", "w", newline="\n") as arquivo_csv:
    escrever = csv.writer(arquivo_csv, delimiter=",", lineterminator="\n")
    count = 0
    while count < len(arquivo_temp):
    	escrever.writerow(arquivo_temp[count])
    	count += 1

del(arquivo_temp[0])
arquivo_temp = list(map(list, zip(*arquivo_temp)))
with open("data/Arquivo_Temp.csv", "w", newline="\n") as arquivo_csv:
    escrever = csv.writer(arquivo_csv, delimiter=",", lineterminator="\n")
    count = 0
    while count < len(arquivo_temp):
    	escrever.writerow(arquivo_temp[count])
    	count += 1

os.system("csv2arff data/Arquivo_Temp.csv data/ArquivoARFF.arff")
os.system("rm data/Arquivo_Temp.csv")

# loader = Loader(classname="weka.core.converters.ArffLoader")
# data = loader.load_file("data/ArquivoARFF.arff")
# data.class_is_last()
# os.system("rm data/ArquivoARFF.arff")

# #InfoGainAttributeEval
# search_InfoGainAttributeEval = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
# evaluator_InfoGainAttributeEval = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
# attsel_InfoGainAttributeEval = AttributeSelection()
# attsel_InfoGainAttributeEval.search(search_InfoGainAttributeEval)
# attsel_InfoGainAttributeEval.evaluator(evaluator_InfoGainAttributeEval)
# attsel_InfoGainAttributeEval.select_attributes(data)
# lista_InfoGainAttributeEval = attsel_InfoGainAttributeEval.selected_attributes

# """ Salvar arquivo com a ordenação das features
# arquivo = open("data/AdverseEvent_InfoGainAttributeEval.txt", "w")
# arquivo.write(str(attsel_InfoGainAttributeEval.results_string))
# arquivo.close()
# """

# #ReliefFAttributeEval
# search_ReliefFAttributeEval = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
# evaluator_ReliefFAttributeEval = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval", options=["-M", "-1", "-D", "1", "-K", "10"])
# attsel_ReliefFAttributeEval = AttributeSelection()
# attsel_ReliefFAttributeEval.search(search_ReliefFAttributeEval)
# attsel_ReliefFAttributeEval.evaluator(evaluator_ReliefFAttributeEval)
# attsel_ReliefFAttributeEval.select_attributes(data)
# lista_ReliefFAttributeEval = attsel_ReliefFAttributeEval.selected_attributes

# """ Salvar arquivo com a ordenação das features
# arquivo = open("data/AdverseEvent_ReliefFAttributeEval.txt", "w")
# arquivo.write(str(attsel_ReliefFAttributeEval.results_string))
# arquivo.close()
# """

# #CorrelationAttributeEval
# search_CorrelationAttributeEval = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
# evaluator_CorrelationAttributeEval = ASEvaluation(classname="weka.attributeSelection.CorrelationAttributeEval")
# attsel_CorrelationAttributeEval = AttributeSelection()
# attsel_CorrelationAttributeEval.search(search_CorrelationAttributeEval)
# attsel_CorrelationAttributeEval.evaluator(evaluator_CorrelationAttributeEval)
# attsel_CorrelationAttributeEval.select_attributes(data)
# lista_CorrelationAttributeEval = attsel_CorrelationAttributeEval.selected_attributes

# """ Salvar arquivo com a ordenação das features
# arquivo = open("data/AdverseEvent_CorrelationAttributeEval.txt", "w")
# arquivo.write(str(attsel_CorrelationAttributeEval.results_string))
# arquivo.close()
# """

# jvm.stop()


# listaTOP = []
# count = 0

# """ Somente as features que aparecem nas 3 listas
# while len(listaTOP) <= TOP - 1:
# 	if lista_InfoGainAttributeEval[count] in lista_ReliefFAttributeEval[0:TOP*3] and lista_InfoGainAttributeEval[count] in lista_CorrelationAttributeEval[0:TOP*3]:
# 		listaTOP.append(lista_InfoGainAttributeEval[count] + 1)
# 		count += 1
# 	else:
# 		count += 1
# """

# # Features em 2 ou 3 listas
# while count <= TOP - 1:
# 	if lista_InfoGainAttributeEval[count] in lista_ReliefFAttributeEval[0:TOP - 1] or lista_InfoGainAttributeEval[count] in lista_CorrelationAttributeEval[0:TOP - 1]:
# 		listaTOP.append(lista_InfoGainAttributeEval[count] + 1)
# 		count += 1
# 	else:
# 		count += 1

# count = 0

# while count <= TOP - 1:
# 	if lista_ReliefFAttributeEval[count] in lista_CorrelationAttributeEval[0:TOP - 1] and not lista_ReliefFAttributeEval[count] in listaTOP:
# 		listaTOP.append(lista_InfoGainAttributeEval[count] + 1)
# 		count += 1
# 	else:
# 		count += 1

# lista_original = []

# # Arquivo original de entrada
# with open("data/ArquivoEntrada.csv", "r") as arquivo_csv:
#     leitor = csv.reader(arquivo_csv)
#     for coluna in leitor:
#         lista_original.append(coluna)

# os.system("rm data/ArquivoEntrada.csv")

# # Arquivo de saída com as features selecionadas para o RF
# with open("data/Arquivo_TOPFeatures.csv", "w", newline="\n") as arquivo_csv:
#     escrever = csv.writer(arquivo_csv, delimiter=",", lineterminator="\n")
#     escrever.writerow(lista_original[0])
#     count_row = 0
#     while count_row <= len(listaTOP) - 1:
#     	escrever.writerow(lista_original[listaTOP[count_row]])
#     	count_row += 1
#     escrever.writerow(lista_original[-1])


# df = pd.read_csv(r"data/Arquivo_TOPFeatures.csv").T
# df.columns = df.iloc[0]
# df.drop(['Probes'],axis=0,inplace=True)
# X = df.drop(['Class'],axis=1)
# genes = X.columns
# y = df['Class']
# lb = LabelBinarizer()
# y = lb.fit_transform(y).ravel()
# df

# os.system("rm data/Arquivo_TOPFeatures.csv")

# trees = 100
# model = RandomForestClassifier(n_estimators=trees)
# model.fit(X, y)
# importance = model.feature_importances_
# index = pd.DataFrame()
# index['Index'] = importance
# index.index = genes
# index.to_csv(f"data/Top{TOP}features_{trees}Trees_.csv")

# print("OK")