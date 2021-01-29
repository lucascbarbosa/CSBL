from weka.core import jvm
from weka.core.converters import Loader, Saver
from weka.core import converters
import csv2arff as arff
import os

jvm.start()

data_dir = "data/"

#os.system("csv2arff data/RNASeq_AntiBody.csv data/RNASeq_AntiBody.arff")

loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(data_dir + "RNASeq_AntiBody.arff")
data.class_is_last()

#InfoGainAttributeEval
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
evaluator = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)

arquivo = open("data/AntiBody_InfoGainAttributeEval.txt", "w")
arquivo.write(str(attsel.results_string))
arquivo.close()

#ReliefFAttributeEval
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
evaluator = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval", options=["-M", "-1", "-D", "1", "-K", "10"])
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)

arquivo = open("data/AntiBody_ReliefFAttributeEval.txt", "w")
arquivo.write(str(attsel.results_string))
arquivo.close()

#CorrelationAttributeEval
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
evaluator = ASEvaluation(classname="weka.attributeSelection.CorrelationAttributeEval")
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)

arquivo = open("data/AntiBody_CorrelationAttributeEval.txt", "w")
arquivo.write(str(attsel.results_string))
arquivo.close()

jvm.stop()

print("OK")