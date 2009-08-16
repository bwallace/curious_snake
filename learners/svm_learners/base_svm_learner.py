'''
	Byron C Wallace and Subie Patel
	Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
	Curious Snake: Active Learning in Python
	base_svm_learner.py
	---
	
	A base class for active learners using Support Vector Machines (SVMs). Uses (a modified version of) the
	libsvm library, Copyright (c) 2000-2008 Chih-Chung Chang and Chih-Jen Lin. 
	
	Subclass this if you want to implement a different active learning strategy with SVMs (see the random_svm_learner and 
	simple_svm_learner modules).
'''

import os
import sys
import pdb

#
# Here we explicitly append the path to libsvm; is there a better way to do this?
#
path_to_libsvm = os.path.join(os.getcwd(), "learners", "svm_learners", "libsvm", "python")
sys.path.append(path_to_libsvm)
import svm
from svm import *
path_to_base_learner = os.path.join(os.getcwd(), "learners")
sys.path.append(path_to_base_learner)
import base_learner
from base_learner import BaseLearner

class BaseSVMLearner(BaseLearner):
    
    def __init__(self, unlabeled_datasets = [], models = None, undersample_before_eval=False):
        BaseLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, 
                                undersample_before_eval=undersample_before_eval)
        # params correspond to each of the respective models (one if we're in a single feature space)
        # these specify things like what kind of kernel to use. here we just use the default, but
        # *you'll probably want to overwrite this* in your subclass. see the libsvm doc for more information (in particular,
        # svm_test.py is helpful).
        self.params = [svm_parameter()  for d in unlabeled_datasets]
        self.div_hash = {}
        
        
    def rebuild_models(self, for_eval=False):
        ''' Rebuilds all models over the current labeled datasets. '''
        datasets = self.labeled_datasets
        # we assume here -- as it's the typical thing to do --
        # that if you are undersampling, you only want to do so 
        # before an evaluation, rather than *during* active learning
        if self.undersample_before_eval and for_eval:
            print "undersampling before building models.."
            datasets = self.undersample_function()
        
        print "training model(s) on %s instances" % len(datasets[0].instances)
        
        self.models = []
        for dataset, param in zip(datasets, self.params):
            samples, labels = dataset.get_samples_and_labels()
            problem = svm_problem(labels, samples)
            self.models.append(svm_model(problem, param))
        print "models rebuilt."

    def _get_dist_from_l(self, model, data, x):
        min_dist = None
        for y in data.instances:
            if not (x.id, y.id) in self.dist_hash:
                self.dist_hash[(x.id, y.id)] = model.compute_dist_between_examples(x.point, y.point)
            if not min_dist or self.dist_hash[(x.id, y.id)] < min_dist:
                min_dist = self.dist_hash[(x.id, y.id)]
        return min_dist


    def _compute_div(self, model, data, x):
        sum = 0.0
        for y in data.instances:
            # have we already computed this?
            if not (x.id, y.id) in self.div_hash:
                # if not, compute the function and add to the hash
                self.div_hash[(x.id, y.id)] = model.compute_cos_between_examples(x.point, y.point)
            sum+= self.div_hash[(x.id, y.id)]
        return sum


    def _compute_cos(self, model_index, x, y):
        ''' 
        computes the cosine between two instances, x and y. note that this memoizes
        (caches) the cosine, to avoid redundant computation.
        '''
        if not (model_index, x.id, y.id) in self.div_hash:
            model = self.models[model_index]
            self.div_hash[(model_index, x.id, y.id)] = model.compute_cos_between_examples(x.point, y.point)
        return self.div_hash[(model_index, x.id, y.id)]
        
    
    def _SIMPLE(self, model, unlabeled_dataset, k):  
        '''
        Implementation of SIMPLE; takes model and dataset to use parametrically.
        Returns selected instance identifiers, as provided by their id fields.
        
        Note that this method lives in this (super) class, rather than, e.g., in 
        simple_svm_learner, because lots of learners use _SIMPLE as one of multiple
        online learning strategies (e.g., brinker (Diverse), wallace (PAL)). Further,
        some sampling strategies (e.g., aggressive undersampling; defined below) require this method.
        Hence we make it more accessible by defining it at the base_svm_learner
        level.
        '''    
        # initially assume k first examples are closest
        k_ids_to_distances = {}
        for x in unlabeled_dataset.instances.values()[:k]:
            k_ids_to_distances[x.id] = model.distance_to_hyperplane(x.point)

        # now iterate over the rest
        for x in unlabeled_dataset.instances.values()[k:]:
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
        
        
    def aggressive_undersample_labeled_datasets(self, k=None):
        '''
        Aggressively undersamples the current labeled datasets; returns a *copy* of the undersampled datasets.
        *Does not mutate the labeled datasets*.
        '''
        feature_space_index = 0
        if self.labeled_datasets and len(self.labeled_datasets) and (len(self.labeled_datasets[0].instances) > 0):
            if not k:
                print "(aggressively) undersampling majority class to equal that of the minority examples"
                # we have to include 'false' minorities -- i.e., instances we've assumed are positives -- because otherwise we'd be cheating
                k = self.labeled_datasets[feature_space_index].number_of_majority_examples() - self.labeled_datasets[0].number_of_minority_examples()
            # we copy the datasets rather than mutate the class members.
            copied_datasets = [d.copy() for d in self.labeled_datasets]
            if k < self.labeled_datasets[0].number_of_majority_examples() and k > 0:
                print "removing %s majority instances. there are %s total majority examples in the dataset." % \
                                    (k, self.labeled_datasets[0].number_of_majority_examples())

                # get the majority examples; find those closeset to the hyperplane (via the SIMPLE method)
                # and return them.
                majority_examples = list(self.labeled_datasets[feature_space_index].get_majority_examples())
                majority_ids = [inst.id for inst in majority_examples]
                majority_dataset = dataset.dataset(instances=dict(zip(majority_ids, majority_examples)))
                removed_these = self._SIMPLE(self.models[feature_space_index], majority_dataset, int(k))
                # if there is more than one feature-space, remove the same instances from the remaining spaces (sets)
                for labeled_dataset in copied_datasets:
                    # now remove them from the corresponding sets
                    labeled_dataset.remove_instances(removed_these)
        else:
            raise Exception, "No labeled data has been provided!"   
        return copied_datasets
        