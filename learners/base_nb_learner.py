#!/usr/bin/env python
# encoding: utf-8
'''
	Byron C Wallace
	Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
	Curious Snake: Active Learning in Python
	base_nb_learner.py
	---

'''

import sys
import os
import pdb
import numpy
import base_learner
from base_learner import BaseLearner
import naive_bayes

class NBLearner(BaseLearner):
    def __init__(self, unlabeled_datasets):
		BaseLearner.__init__(self, unlabeled_datasets=unlabeled_datasets)
		
    
    def _datasets_to_matrices(self, datasets):
	    ''' 
	    Returns the datasets in a format palatable to the bayesian lin. reg. library,
	    i.e., as a matrix of observations (phi) and a vector of labels (t)
	    '''
	    phi, t = [], []
	    for instance in datasets:
	        phi.append(instance[:-1])
	        pdb.set_trace()
	        t.append(instance[-1])

	    return(numpy.array(phi), numpy.array(t))

    def rebuild_models(self):
        ''' Rebuilds all models over the current labeled datasets. '''    
        if self.undersample_first:
            print "undersampling before building models.."
            datasets = self.undersample_labeled_datasets()

        datasets = self._datasets_to_matrices(datasets)