import pdb
import base_learner
from base_learner import BaseLearner
from base_learner import *

self.query_function = 

def __init__(self, unlabeled_datasets = [], models=None):
    #
    # call the BaseLearner constructor to initialize various globals and process the
    # datasets, etc.; of course, these can subsequently be overwritten.
    BaseLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models)
    
    # ovewrite svm parameters here 
    self.params = [svm_parameter()  for d in unlabeled_datasets]
    
    print "switching query function to RANDOM"
    #
    # Here we switch the query function to randomly sampling. Note that 
    # this function actually lives in the base_learner parent class, because
    # it may very well be useful for other learners to request ids for random
    # unlabeled examples
    #
    self.query_function = self.get_random_unlabeled_ids # 
    self.name = "Random"