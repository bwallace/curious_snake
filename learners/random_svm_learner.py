import pdb
import base_svm_learner
from base_svm_learner import BaseSVMLearner
from base_svm_learner import *

class RandomLearner(BaseSVMLearner):
    def __init__(self, unlabeled_datasets = [], models=None, undersample_before_eval=False):
        #
        # call the BaseLearner constructor to initialize various globals and process the
        # datasets, etc.; of course, these can subsequently be overwritten.
        BaseSVMLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models,
                                    undersample_before_eval=undersample_before_eval)

        # ovewrite svm parameters here 
        self.params = [svm_parameter()  for d in unlabeled_datasets]
    
        print "switching query function to RANDOM"
        #
        # Here we switch the query function to randomly sampling. Note that 
        # this function actually lives in the base_learner parent class, because
        # it may very well be useful for other learners to request ids for random
        # unlabeled examples
        #
        self.query_function = self.get_random_unlabeled_ids 
        self.name = "Random"