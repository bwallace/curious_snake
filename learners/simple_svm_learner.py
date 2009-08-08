import pdb
import base_svm_learner
from base_svm_learner import BaseSVMLearner
from base_svm_learner import *

class SimpleLearner(BaseSVMLearner):
    
    def __init__(self, unlabeled_datasets = [], models=None):
        #
        # call the BaseLearner constructor to initialize various globals and process the
        # datasets, etc.; of course, these can subsequently be overwritten.
        BaseSVMLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models)
        #super(SimpleLearner, self).__init__()
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
        # just uses the first feature space, if there are multiple it ignores the rest.
        model = self.models[0] 
        dataset = self.unlabeled_datasets[0]
        return self._SIMPLE(model, dataset, k)