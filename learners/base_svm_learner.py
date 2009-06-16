# append path to the svmlib
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
        #super(BaseSVMLearner, self).__init__(unlabeled_datasets = unlabeled_datasets)
        # params correspond to each of the respective models (one if we're in a single feature space)
        # these specify things like what kind of kernel to use. here we just use the default, but
        # *you'll probably want to overwrite this* in your subclass. see the libsvm doc for more information (in particular,
        # svm_test.py is helpful).
        self.params = [svm_parameter()  for d in unlabeled_datasets]
        self.div_hash = {}
        
    def rebuild_models(self):
        ''' Rebuilds all models over the current labeled datasets. '''    
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
    
    def predict(self, X):
        ''' 
        This defines how we will predict labels for new examples. We use a simple ensemble voting
        strategy if there are multiple feature spaces. If there is just one feature space, this just
        uses the libSVM predict function. 
        '''
        return self.majority_predict(X)
            
            
    def majority_predict(self, X):
        '''
        If there are multiple models built over different feature spaces, this predicts a label for an instance based on the
        majority vote of these classifiers -- otherwise this is simply "predict"
        '''
        votes = []
        if self.models and len(self.models):
            for m,x in zip(self.models, X):
                votes.append(m.predict(x))
            vote_set = list(set(votes))
            count_in_list = lambda x: votes.count(x)
            return vote_set[_arg_max(vote_set, count_in_list)]
        else:
            raise Exception, "No models have been initialized."

    def cautious_predict(self, X):
        '''
        A naive way of combining different models (built over different feature-spaces); if any othe models vote yes, then vote yes.
        When there is only on feature space, this reduces to simply "predict".
        '''
        if self.models and len(self.models):
            return max([m.predict(x) for m,x in zip(self.models, X)])
        else:
            raise Exception, "No models have been initialized."

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
        
def _arg_max(ls, f):
    ''' Returns the index for x in ls for which f(x) is maximal w.r.t. the rest of the list '''
    return_index = 0
    max_val = f(ls[0])
    for i in range(len(ls)-1):
        if f(ls[i+1]) > max_val:
            return_index = i
            max_val = f(ls[i+1])
    return return_index