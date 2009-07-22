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
#
# Here we explicitly append the path to libsvm; is there a better way to do this?
#
import os
import sys
import pdb
path_to_libsvm = os.path.join(os.getcwd(), "learners", "libsvm", "python")
sys.path.append(path_to_libsvm)
import svm
from svm import *
import base_learner
from base_learner import BaseLearner

class BaseSVMLearner(BaseLearner):
    
    def __init__(self, unlabeled_datasets = [], models = None):
        BaseLearner.__init__(self, unlabeled_datasets=unlabeled_datasets)
        # params correspond to each of the respective models (one if we're in a single feature space)
        # these specify things like what kind of kernel to use. here we just use the default, but
        # *you'll probably want to overwrite this* in your subclass. see the libsvm doc for more information (in particular,
        # svm_test.py is helpful).
        self.params = [svm_parameter()  for d in unlabeled_datasets]
        self.div_hash = {}
        
    def rebuild_models(self):
        ''' Rebuilds all models over the current labeled datasets. '''
        dataset = None
        if self.undersample_first:
            print "undersampling before building models.."
            datasets = self.undersample_labeled_datasets()
            print "done."
        else:
            datasets = self.labeled_datasets
        
        print "training model(s) on %s instances" % len(datasets[0].instances)
        self.models = []
        for dataset, param in zip(datasets, self.params):
            samples, labels = dataset.get_samples_and_labels()
            problem = svm_problem(labels, samples)
            self.models.append(svm_model(problem, param))
        print "done."

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


    def _compute_cos(self, model, x, y):
        if not (x.id, y.id) in self.div_hash:
            self.div_hash[(x.id, y.id)] = model.compute_cos_between_examples(x.point, y.point)
        return self.div_hash[(x.id, y.id)]
        