import pdb
import base_learner
from base_learner import BaseLearner
from base_learner import *

class SimpleLearner(BaseLearner):
    
    def __init__(self, unlabeled_datasets = [], models=None):
        #
        # call the BaseLearner constructor to initialize various globals and process the
        # datasets, etc.; of course, these can subsequently be overwritten.
        BaseLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models)
        
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
          
        
    def _SIMPLE(self, model, unlabeled_dataset, k):  
        '''
        Implementation of SIMPLE; takes model and dataset to use parametrically.
        Returns selected instance identifiers, as provided by their id fields.
        '''    
        # initially assume k first examples are closest
        k_ids_to_distances = {}
        for x in unlabeled_dataset.instances[:k]:
            k_ids_to_distances[x.id] = model.distance_to_hyperplane(x.point)
        
        # now iterate over the rest
        for x in unlabeled_dataset.instances[k:]:
            cur_max_id, cur_max_dist = self._get_max_val_key_tuple(k_ids_to_distances)
            x_dist = model.distance_to_hyperplane(x.point)
            if x_dist < cur_max_dist:
                # then x is closer to the hyperplane than the farthest currently observed
                # remove current max entry from the dictionary
                k_ids_to_distances.pop(cur_max_id)
                k_ids_to_distances[x.id] = x_dist
    
        return k_ids_to_distances.keys()        
        
        
    def _get_max_val_key_tuple(self, d):
        keys, values = d.keys(), d.values()
        max_key, max_val = keys[0], values[0]
        for key, value in zip(keys[1:], values[1:]):
            if value > max_val:
                max_key = key
                max_val = value
        return (max_key, max_val)