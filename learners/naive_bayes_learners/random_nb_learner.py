'''
	Byron C Wallace and Subie Patel
	Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
	Curious Snake: Active Learning in Python
	base_nb_learner.py
	---
    
    Random Naive Bayes learner; A naive naive bayes learner, queries for labels at random.
'''
import base_nb_learner
from base_nb_learner import BaseNBLearner

class RandomNBLearner(BaseNBLearner):
    def __init__(self, unlabeled_datasets = [], models=None, undersample_before_eval=False):
        #
        # call the BaseLearner constructor to initialize various globals and process the
        # datasets, etc.; of course, these can subsequently be overwritten.
        BaseNBLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models,
                                    undersample_before_eval=undersample_before_eval)
        
        # use the random query function (i.e., ask for labels at random)
        self.query_function = self.get_random_unlabeled_ids 
        self.name = "Random Naive Bayes"