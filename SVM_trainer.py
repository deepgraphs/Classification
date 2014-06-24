#!/usr/bin/env python
__author__ = "VanKos"

import PyML
from PyML import *
import logging
from os.path import join
__version__ = PyML.__version__
from PyML.containers.vectorDatasets import SparseDataSet
from PyML.containers.vectorDatasets import VectorDataSet
from PyML.classifiers import multi

logging.basicConfig(level=logging.INFO, filename=join('.','testrun.log'),
                    format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('').addHandler(logging.StreamHandler())


# data = VectorDataSet('data/heartSparse.data',labelsColumn = 0)
data = SparseDataSet('data/matrixPyML',labelsColumn = 0)
mc = multi.OneAgainstRest (SVM())
mc.train(data)
r = mc.cv(data, 5)
mc.save("SVM")
# s = SVM()
# s.train(data)
# r = s.cv(data, 5)
# s.save("SVM")
print r
