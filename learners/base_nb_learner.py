#!/usr/bin/env python
# encoding: utf-8
'''
	Byron C Wallace
	Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
	Curious Snake: Active Learning in Python
	base_nb_learner.py
	---
    
    Naive Bayes learner.
'''

import sys
import os
import pdb
import numpy
import base_learner
from base_learner import BaseLearner
import naive_bayes
from naive_bayes import NaiveBayes

#
# TODO implement train
#

class NBLearner(BaseLearner):
    def __init__(self, unlabeled_datasets):
		BaseLearner.__init__(self, unlabeled_datasets=unlabeled_datasets)
		
    def _datasets_to_matrices(self, datasets):
	    ''' Returns the datasets in a format palatable to the naive bayes module.'''
	    dims = []
	    for dataset in datasets:
	        point_maxes = max([max([inst.point.keys()]) for inst in dataset.instances])
	        max_for_dataset = max(point_maxes)
	        dims.append(max_for_dataset)

	    all_instances = []
	    for dataset in datasets:
	        instances = []
	        for dimensionality, instance in zip(dims, dataset.instances):
	            cur_inst = []
	            for x in range(dimensionality):
	                if x not in instance.point.keys():
	                    cur_inst.append(0.0)
	                else:
	                    cur_inst.append(instance.point[x])
                instances.append(cur_inst)
            all_instances.append(instances)
        
        # the labels will be the same for all datasets
	    labels = [instance.label for instance in datasets[0].instances]
	    return (all_instances, labels)
	    
	    
    def rebuild_models(self):
        ''' Rebuilds all models over the current labeled datasets. '''
        datasets = self.labeled_datasets
        if self.undersample_first:
            print "undersampling before building models.."
            datasets = self.undersample_labeled_datasets()

        all_train_sets, labels = self._datasets_to_matrices(datasets)
        self.models = [NB_Model(NaiveBayes.train(training_set, labels)) for training_set in all_train_sets]
    
        
class NB_Model(object):
    '''
    Wraps the Naive Bayes model so that it plays nice with the AL framework. (We need the predict method, in particular).
    '''
    def __init__(nb):
        self.nbayes_model = nb
        
    def predict(self,x):
        return self.nbayes_model.classify(x) 
        
    def prob_dist(self, x):
        return self.nbayes_model.calculate(x)
        
        
        
        