import weka.core.jvm as jvm
import weka.core.converters as converters
import os
import csv
import pandas as pd
import numpy as np
from weka.core.converters import Loader, Saver
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from sklearn.preprocessing import LabelBinarizer 
from collections import Counter
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
	
	lista_InfoGainAttributeEval = list(lista_InfoGainAttributeEval)
	lista_ReliefFAttributeEval = list(lista_ReliefFAttributeEval)
	lista_CorrelationAttributeEval = list(lista_CorrelationAttributeEval)
	lista_InfoGainAttributeEval.remove(len(lista_InfoGainAttributeEval)-1)
	lista_InfoGainAttributeEval.remove(0)
	lista_ReliefFAttributeEval.remove(len(lista_ReliefFAttributeEval)-1)
	lista_ReliefFAttributeEval.remove(0)
	lista_CorrelationAttributeEval.remove(len(lista_CorrelationAttributeEval)-1)
	lista_CorrelationAttributeEval.remove(0)

	# Variável com as features que aparecem em 2 de 3 métodos

	listaTOP = []
	
	if top > len(lista_InfoGainAttributeEval):
		print(f"There is not {top} genes in the dataset. Selecting {len(lista_InfoGainAttributeEval)} genes.")
		lista = lista_InfoGainAttributeEval + lista_ReliefFAttributeEval + lista_CorrelationAttributeEval
	else:
		lista = list(lista_InfoGainAttributeEval[:top-1])+list(lista_ReliefFAttributeEval[:top-1])+list(lista_CorrelationAttributeEval[:top-1])
	
	counts = Counter(lista)
	for el in counts.keys():
		if counts[el] >=2:
			listaTOP.append(el-1)

	listaTOP = np.array(listaTOP)

	if file_format == 'txt':
		X = pd.read_csv(input_file_X, sep='\t', header=0, index_col= 0).T.astype(np.float64).round(9)
		X = X[X.columns[listaTOP]]
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
		X = pd.read_csv(input_file_X, header=0, index_col= 0).T.astype(np.float64).round(9)
		X = X[X.columns[listaTOP]]
		y = pd.read_csv(input_file_y, header=0, index_col= 0)
		y = y[coly]
		y = y.dropna()
		X = X.loc[np.intersect1d(X.index.values,y.index.values)]
		y = y.loc[np.intersect1d(X.index.values,y.index.values)]
		# lb = LabelBinarizer()
		# y = lb.fit_transform(y).ravel().astype(int)
		X['Class'] = y.values
		X.to_csv(top_features_file, header=True, index=True)