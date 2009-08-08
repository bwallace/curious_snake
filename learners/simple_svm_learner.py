import pdb
import random

import base_svm_learner
from base_svm_learner import BaseSVMLearner
from base_svm_learner import *

class SimpleLearner(BaseSVMLearner):
    
    def __init__(self, unlabeled_datasets = [], models=None, undersample_before_eval=False):
        #
        # call the BaseLearner constructor to initialize various globals and process the
        # datasets, etc.; of course, these can subsequently be overwritten.
        BaseSVMLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models, 
                                                    undersample_before_eval=undersample_before_eval)

        # ovewrite svm parameters here 
        self.params = [svm_parameter()  for d in unlabeled_datasets]
        
        print "switching query function to SIMPLE!"
        #
        # most importantly we change the query function to SIMPLE here
        #
        self.query_function = self.SIMPLE
        self.name = "SIMPLE"
        
        
    def SIMPLE(self, k):
        '''
        Returns the instance numbers for the k unlabeled instances closest the hyperplane.
        '''
        feature_space_index = random.randint(0, len(self.models)-1)
        model = self.models[feature_space_index] 
        dataset = self.unlabeled_datasets[feature_space_index]
        return self._SIMPLE(model, dataset, k)