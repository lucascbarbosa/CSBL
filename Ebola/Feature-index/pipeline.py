# import weka.jvm as jvm
# import weka.core.converters as converters
import csv2arff as arff
import os
import csv
# from weka.core.converters import Loader, Saver
# from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection

jvm.start()

# Código p/ transformar o arquivo .csv para a análise
#os.system("csv2arff data/RNASeq_AdverseEvent.csv data/RNASeq_AdverseEvent.arff")

loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("data/RNASeq_AdverseEvent.arff")
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
TOP = 100

listaTOP = []
count = 0

""" Somente as features que aparecem nas 3 listas
while len(listaTOP) < TOP:
	if lista_InfoGainAttributeEval[count] in lista_ReliefFAttributeEval[0:TOP*3] and lista_InfoGainAttributeEval[count] in lista_CorrelationAttributeEval[0:TOP*3]:
		listaTOP.append(lista_InfoGainAttributeEval[count] + 1)
		count += 1
	else:
		count += 1
"""

# Features em 2 ou 3 listas
for count in range(TOP):
	if lista_InfoGainAttributeEval[count] in lista_ReliefFAttributeEval[:TOP - 1] or lista_InfoGainAttributeEval[count] in lista_CorrelationAttributeEval[:TOP - 1]:
		listaTOP.append(lista_InfoGainAttributeEval[count] + 1)


for count in range(TOP):
	if lista_ReliefFAttributeEval[count] in lista_CorrelationAttributeEval[:TOP - 1] and not lista_ReliefFAttributeEval[count] in listaTOP:
		listaTOP.append(lista_InfoGainAttributeEval[count] + 1)

lista_original = []

# Arquivo original de entrada
with open("Input_data/RNASeq_AdverseEvent.csv", "r") as arquivo_csv:
    leitor = csv.reader(arquivo_csv)
    for coluna in leitor:
        lista_original.append(coluna)

# Arquivo de saída com as features selecionadas para o RF
with open("data/RNASeq_AdverseEvent_TOP100.csv", "w", newline="\n") as arquivo_csv:
    escrever = csv.writer(arquivo_csv, delimiter=",", lineterminator="\n")
    escrever.writerow(lista_original[0])
    count_row = 0
    while count_row <= len(listaTOP) - 1:
    	escrever.writerow(lista_original[listaTOP[count_row]])
    	count_row += 1
    escrever.writerow(lista_original[-1])

print("OK")