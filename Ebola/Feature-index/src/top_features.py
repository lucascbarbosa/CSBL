import weka.core.jvm as jvm
import weka.core.converters as converters
import os
import csv
import pandas as pd
import numpy as np
from weka.core.converters import Loader, Saver
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from sklearn.preprocessing import LabelBinarizer 
import warnings
warnings.filterwarnings("ignore")

def top_features(input_file_X,input_file_y,filtered_input_file,top_features_file,coly,file_format,top):
	jvm.start()

	loader = Loader(classname="weka.core.converters.ArffLoader")
	path_arff = str(filtered_input_file)[:-4]+'.arff'
	data = loader.load_file(path_arff)
	data.class_is_last()

	#InfoGainAttributeEval
	search_InfoGainAttributeEval = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
	evaluator_InfoGainAttributeEval = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
	attsel_InfoGainAttributeEval = AttributeSelection()
	attsel_InfoGainAttributeEval.search(search_InfoGainAttributeEval)
	attsel_InfoGainAttributeEval.evaluator(evaluator_InfoGainAttributeEval)
	attsel_InfoGainAttributeEval.select_attributes(data)
	lista_InfoGainAttributeEval = attsel_InfoGainAttributeEval.selected_attributes

	""" Salvar arquivo com a ordenação das features
	arquivo = open("data/AdverseEvent_InfoGainAttributeEval.txt", "w")
	arquivo.write(str(attsel_InfoGainAttributeEval.results_string))
	arquivo.close()
	"""

	#ReliefFAttributeEval
	search_ReliefFAttributeEval = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
	evaluator_ReliefFAttributeEval = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval", options=["-M", "-1", "-D", "1", "-K", "10"])
	attsel_ReliefFAttributeEval = AttributeSelection()
	attsel_ReliefFAttributeEval.search(search_ReliefFAttributeEval)
	attsel_ReliefFAttributeEval.evaluator(evaluator_ReliefFAttributeEval)
	attsel_ReliefFAttributeEval.select_attributes(data)
	lista_ReliefFAttributeEval = attsel_ReliefFAttributeEval.selected_attributes

	""" Salvar arquivo com a ordenação das features
	arquivo = open("data/AdverseEvent_ReliefFAttributeEval.txt", "w")
	arquivo.write(str(attsel_ReliefFAttributeEval.results_string))
	arquivo.close()
	"""

	#CorrelationAttributeEval
	search_CorrelationAttributeEval = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
	evaluator_CorrelationAttributeEval = ASEvaluation(classname="weka.attributeSelection.CorrelationAttributeEval")
	attsel_CorrelationAttributeEval = AttributeSelection()
	attsel_CorrelationAttributeEval.search(search_CorrelationAttributeEval)
	attsel_CorrelationAttributeEval.evaluator(evaluator_CorrelationAttributeEval)
	attsel_CorrelationAttributeEval.select_attributes(data)
	lista_CorrelationAttributeEval = attsel_CorrelationAttributeEval.selected_attributes

	""" Salvar arquivo com a ordenação das features
	arquivo = open("data/AdverseEvent_CorrelationAttributeEval.txt", "w")
	arquivo.write(str(attsel_CorrelationAttributeEval.results_string))
	arquivo.close()
	"""

	jvm.stop()

	# Variável com o valor da quantidade de features

	listaTOP = []
	count = 0

	""" Somente as features que aparecem nas 3 listas
	while len(listaTOP) <= TOP - 1:
		if lista_InfoGainAttributeEval[count] in lista_ReliefFAttributeEval[0:TOP*3] and lista_InfoGainAttributeEval[count] in lista_CorrelationAttributeEval[0:TOP*3]:
			listaTOP.append(lista_InfoGainAttributeEval[count] + 1)
			count += 1
		else:
			count += 1
	"""

	# Features em 2 ou 3 listas
	while count <= top - 1:
		if lista_InfoGainAttributeEval[count] in lista_ReliefFAttributeEval[0:top - 1] or lista_InfoGainAttributeEval[count] in lista_CorrelationAttributeEval[0:top - 1]:
			listaTOP.append(lista_InfoGainAttributeEval[count] + 1)
			count += 1
		else:
			count += 1

	count = 0

	while count <= top - 1:
		if lista_ReliefFAttributeEval[count] in lista_CorrelationAttributeEval[0:top - 1] and not lista_ReliefFAttributeEval[count] in listaTOP:
			listaTOP.append(lista_InfoGainAttributeEval[count] + 1)
			count += 1
		else:
			count += 1
	if file_format == 'txt':
		X = pd.read_csv(input_file_X, sep='\t', header=0, index_col= 0).T
		X = X.iloc[:,listaTOP]
		y = pd.read_csv(input_file_y, sep='\t', header=0, index_col= 0)
		y = y[coly]
		y = y.dropna()
		X = X.loc[np.intersect1d(X.index.values,y.index.values)]
		y = y.loc[np.intersect1d(X.index.values,y.index.values)]
		# lb = LabelBinarizer()
		# y = lb.fit_transform(y).ravel().astype(int)
		X['Class'] = y.values
		X.to_csv(top_features_file,sep='\t', header=True, index=True)

	if file_format == 'csv':
		X = pd.read_csv(input_file_X, header=0, index_col= 0).T
		X = X.iloc[:,listaTOP]
		y = pd.read_csv(input_file_y, header=0, index_col= 0)
		y = y[coly]
		y = y.dropna()
		X = X.loc[np.intersect1d(X.index.values,y.index.values)]
		y = y.loc[np.intersect1d(X.index.values,y.index.values)]
		# lb = LabelBinarizer()
		# y = lb.fit_transform(y).ravel().astype(int)
		X['Class'] = y.values
		X.to_csv(top_features_file, header=True, index=True)