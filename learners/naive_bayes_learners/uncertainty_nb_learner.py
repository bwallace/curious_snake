import pdb
import random
import base_nb_learner
from base_nb_learner import BaseNBLearner

class UncertaintyNBLearner(BaseNBLearner):
    
    def __init__(self, unlabeled_datasets = [], models=None, undersample_before_eval=False):
        #
        # call the BaseLearner constructor to initialize various globals and process the
        # datasets, etc.; of course, these can subsequently be overwritten.
        BaseNBLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models,
                                    undersample_before_eval=undersample_before_eval)
        
        # set the query function to uncertainty sampling
        self.query_function = self.uncertainty_sample
        self.name = "Uncertain Naive Bayes"
		
	
    def uncertainty_sample(self, k):
	    '''
	    Pick the k examples we're least certain about
	    '''
	    dataset_index = random.randint(0,len(self.unlabeled_datasets)-1)
	        
	    # assume we are least certain about the first k, cache how certain we are
	    ids_to_certainties = {}
	    for inst_id, inst in self.unlabeled_datasets[dataset_index].instances.items()[:k]:
	        ids_to_certainties[inst_id] = abs(.5 - self._closest_to(self.models[dataset_index].prob_dist(inst.point).values(), 
	                                                        .5))
	        
	    # now check the remaining examples to see if we're more uncertain about any of them
	    for inst_id, inst in self.unlabeled_datasets[dataset_index].instances.items()[k:]:
	        # find the example we're most certain about currently; this is what we'll
	        # compare the rest of the pool to 
	        highest_id, highest_certainty = ids_to_certainties.items()[0]
	        for inst_id, certainty in ids_to_certainties.items()[1:]:
	            if certainty > highest_certainty:
	                highest_id, highest_certainty = inst_id, certainty
	                
	        cur_certainty =  abs(.5 - self._closest_to(self.models[dataset_index].prob_dist(inst.point).values(), .5))
	        # now check if we're less certain about this example than the example
	        # we're currently most certain about; if the distance from .5 is less
	        # than the most certain one, we swap them (we're less confident in this case)
            if cur_certainty < highest_certainty:
	            ids_to_certainties.pop(highest_id)
	            ids_to_certainties[inst_id] = cur_certainty
	    return ids_to_certainties.keys()
        
	
    def _closest_to(self, vals, x):
	    ''' returns the value in vals closest to x '''
	    closest = vals[0]
	    closest_dist = abs(x-closest)
	    for val in vals[1:]:
	        cur_dist = abs(x-closest)
	        if cur_dist < closest_dist:
	            closest = x
	            closest_dist = cur_dist
	    return closest
	    

	    
