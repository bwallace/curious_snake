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


class NBLearner(BaseLearner):
    '''
    Base Naive Bayes learner. 
    '''
    def __init__(self, unlabeled_datasets):
		BaseLearner.__init__(self, unlabeled_datasets=unlabeled_datasets)
		self.query_function = self.get_random_unlabeled_ids 
		self.name = "Random Naive Bayes"
        
    def _datasets_to_matrices(self, datasets):
	    ''' 
	    Returns the datasets in a format palatable to the naive bayes module.
	    
	    Generally, this will need to be done when the format used to train the model
	    differs from the the ``libSVM style'' format used by curious snake (which is
	    a sparse format).
	    '''
	    dims = []
	    for dataset in datasets:
	        point_maxes = max([max([inst.point.keys()]) for inst in dataset.instances])
	        max_for_dataset = max(point_maxes)
	        dims.append(max_for_dataset)

	    all_instances = []
	    for dimensionality, dataset in zip(dims, datasets):
	        instances = []
	        i = 0
	        for instance in dataset.instances:
	            print "i: %s" % i
	            print instances
	            i+=1
	            cur_inst = []
	            for x in range(dimensionality):
	                if x not in instance.point.keys():
	                    cur_inst.append(0.0)
	                else:
	                    cur_inst.append(instance.point[x])
	            instances.append(cur_inst)
	            print "\nblegh!"
	            print instances
	            #print "instance! %s" % str(instance)
	            #print "cur_inst! %s" % str(cur_inst)
	            #print "instances! %s" % str(instances)
            
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
        self.models = [NB_Model(naive_bayes.train(training_set, labels)) for training_set in all_train_sets]

        
class NB_Model(object):
    '''
    Wraps the Naive Bayes model so that it plays nice with the AL framework. (We need the predict method, in particular).
    '''
    def __init__(self,nb):
        self.nbayes_model = nb
        
    def predict(self,x):
        return naive_bayes.classify(self.nbayes_model, self._map_x(x))
        
    def prob_dist(self, x):
        return naive_bayes.calculate(self.nbayes_model, self._map_x(x))
        
    def _map_x(self, x):
        x_prime = {}
        # first, make sure x has the same dimensionality as the model
        for dim in range(self.nbayes_model.dimensionality):
            if dim not in x:
                x_prime[dim] = 0.0
            else:
                x_prime[dim] = x[dim]
        return x_prime
 
        