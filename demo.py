import csv

import numpy as np

from test import TestSVM
from ensemble import Ensemble
from ensemblesvm import EnsembleSVM
from svm import SVM
from chsvm import ChSVM
from naive import MNB
from dictionary import Dictionary
from logreg import LogReg
from chlogreg import ChLogReg
from dictgbc import DictGBC
from dictknn import DictKNN
from dictrfc import DictRFC
from auc import calculate_auc
from nlp_dict import NLPDict
from nlpdictch import NLPDictCh
from sgdc import SGDC
import sys
import time
import signal

start = time.time()
texts = []
classes = []
csvr = csv.reader(open('./dataset/train.csv', 'rb'), delimiter=',', quotechar='"')
csvr.next()
for row in csvr:
    texts.append(row[2].decode('utf8'))
    classes.append(int(row[0]))
    
#nlpdict = NLPDict(texts=texts)
#nlpdictch = NLPDictCh(texts=texts) 
    
models = ( \
    (SVM, ()), \
#    (MNB, ()), \
    (Dictionary, ()), \
#    (LogReg, (None, 1, 1.0)), \
#    (DictGBC, ()), \
    (ChSVM, ()), \
#    (DictRFC, ()), \
)

nn_params = {'epochs': int(sys.argv[1]), 'structure': [3, 1]}
#n = Ensemble(texts, classes, nn_params, models)
 
# m1 = ChSVM(texts, classes)
# m2 = Dictionary(texts, classes)
 
# texts = []
# classes = []
# csvr = csv.reader(open('./dataset/test_with_solutions.csv', 'rb'), delimiter=',', quotechar='"')
# csvr.next()
# for row in csvr:
#     texts.append(row[2].decode('utf8'))
#     classes.append(int(row[0]))
# #results = n.classify(texts)
# #results[results<0] = 0
# #print calculate_auc(classes, results)
# r1 = np.array(m1.classify(texts))
# print calculate_auc(classes, r1)
# r2 = np.array(m2.classify(texts))
# print calculate_auc(classes, r2)
# r = (1.2*r1 + 0.8*r2) / 2
# r[r>1] = 1
# r[r<0] = 0
# print calculate_auc(classes, r)
  
#print TestSVM.test_model(texts, classes, models[-1])
#print TestSVM.test(texts, classes, models, nn_params)
n = Ensemble(texts, classes, nn_params, models)
end = time.time()
# print "training time="
# print end-start

start = time.time()

# evaluate the classfier on verification dataset
texts = []
inp = raw_input()
while inp:
	texts.append(inp.decode('utf8'))
	inp = raw_input()

results = n.classify(texts)

print results



