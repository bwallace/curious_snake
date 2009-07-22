import pdb
import base_nb_learner
from base_nb_learner import BaseNBLearner

class UncertaintyNBLearner(BaseNBLearner):
    
    def __init__(self, unlabeled_datasets = [], models=None):
        #
        # call the BaseLearner constructor to initialize various globals and process the
        # datasets, etc.; of course, these can subsequently be overwritten.
        BaseNBLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models)
        
        # set the query function to uncertainty sampling
        pdb.set_trace()
        self.query_function = self.uncertainty_sample
        self.name = "Uncertain Naive Bayes"
		
	
    def uncertainty_sample(self, k):
	    '''
	    Pick the k examples we're least certain about
	    '''
	    pdb.set_trace()
	    dataset_index = 0
	    #
	    # TODO now return the k most 
	    #
	    for x in self.unlabeled_datasets[dataset_index].instances:
	        prob_dist = models[dataset_index].prob_dist(x.point)
	        
	    # i believe the prob_dist is on the log scale -- exp !
	    pdb.set_trace()
