'''    
    Byron C Wallace
    Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
    Curious Snake
    base_learner.py
    --
    This module contains the BaseLearner class, which you can subclass  to implement your own 
    (pool-based) active learning strategy. BaseLearner itself can actually be used directly; it implements
    the 'random' strategy, i.e., it picks examples for the expert to label at random. 
'''

import pdb
import os
import sys
import random
import math
import dataset 
import numpy

class BaseLearner(object):
    '''
    Base learner class. Sub-class this object to implement your own learning strategy. 
    
    Repeating the comment in curious_snake.py, Curious Snake was originally written for a scenario in which multiple feature spaces
    were being exploited, thus pluralizing may of the attributes in this class. For example, 
    *lists* of unlabeled_datasets and models are kept. If you only have one feature space that you're interested
     in, as is often the case, simply pass around unary lists.  
    ''' 

    def __init__(self, unlabeled_datasets = None, models = None, undersample_before_eval = False):
        '''
        @params:
        --
        unlabeled_datasets -- a list of Dataset objects
        models -- a list of model objects to be used in classification. usually None, as they've yet
                    to be built.
        undersample_before_eval -- if True, datasets will be undersampled (i.e., the the number of
                                   examples from each classes will be made equal) before the final
                                    classifier is built. see, e.g., Japkowicz:
                                    "The Class Imbalance Problem: Significance and Strategies"
        '''
        if isinstance(unlabeled_datasets, dataset.Dataset):
            # then a single data file was passed in
            unlabeled_datasets  = [unlabeled_datasets]
            
        self.unlabeled_datasets = unlabeled_datasets or []
        # initialize empty labeled datasets (i.e., all data is unlabeled to begin with)
        # note that we give the labeled dataset the same name as the corresponding
        # unlabeled dataset
        self.labeled_datasets = [dataset.Dataset(name=d.name) for d in unlabeled_datasets]

        self.models = models
        self.undersample_before_eval = undersample_before_eval 
        # arbitrary re-sampling functions can be plugged in here
        self.undersample_function = self.undersample_labeled_datasets if undersample_before_eval else None

        self.query_function = self.base_q_function # throws exception if not overridden 
        self.name = "Base"
        self.description = ""

        # default prediction function; only important if you're aggregating multiple feature spaces (see 
        # cautious_predict function documentation)
        self.predict = self.majority_predict
        self.rebuild_models_at_each_iter = True # if this is false, the models will not be rebuilt after each round of active learning

 
    
    def active_learn(self, num_examples_to_label, batch_size=5):
        '''
        Core active learning routine. Here the learner uses its query function to select a number of examples 
        (num_to_label_at_each_iteration) to label at each step, until the total number of examples requested 
        (num_examples_to_label) has been labeled. The models will be updated at each iteration.
        '''
        labeled_so_far = 0
        while labeled_so_far < num_examples_to_label:
            example_ids_to_label = self.query_function(batch_size)
            # now remove the selected examples from the unlabeled sets and put them in the labeled sets.
            # if not ids are returned -- ie., if a void query_function is used --
            # it is assumed the query function took care of labeling the examples selected. 
            if example_ids_to_label:
                self.label_instances_in_all_datasets(example_ids_to_label)

            if self.rebuild_models_at_each_iter:
                self.rebuild_models()   

            labeled_so_far += batch_size

        self.rebuild_models()
            
    
    def predict(self, X):
        ''' 
        This defines how we will predict labels for new examples. We use a simple ensemble voting
        strategy if there are multiple feature spaces. If there is just one feature space, this just
        uses the 'predict' function of the model.
        '''
        return self.majority_predict(X)


    def majority_predict(self, X):
        '''
        If there are multiple models built over different feature spaces, this predicts a label for an instance based on the
        majority vote of these classifiers -- otherwise this is simply "predict"
        '''
        votes = []
        if self.models and len(self.models) > 0:
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
                
    def base_q_function(self, k):
        ''' overwite this method with query function of choice (e.g., SIMPLE) '''
        raise Exception, "no query function provided!"

                                 
    def label_all_data(self):
        '''
        Labels all the examples in the training set
        '''
        inst_ids = [inst.id for inst in self.unlabeled_datasets[0].instances]
        self.label_instances_in_all_datasets(inst_ids)
        
        
    def label_instances(self, inst_ids):
        ''' Just an overloaded name for label_instances_in_all_datasets '''
        self.label_instances_in_all_datasets(inst_ids)
        
    def label_instances_in_all_datasets(self, inst_ids):
        '''
        Removes the instances in inst_ids (a list of instance numbers to 'label') from the unlabeled dataset(s) and places
        them in the labeled dataset(s). These will subsequently be used in training models, thus this simulates 'labeling'
        the instances.
        '''
        for unlabeled_dataset, labeled_dataset in zip(self.unlabeled_datasets, self.labeled_datasets):
            labeled_dataset.add_instances(unlabeled_dataset.remove_instances(inst_ids))  
    

    def pick_balanced_initial_training_set(self, k):
        '''
        Picks k + and k - examples at random for bootstrap set.
        '''
        minority_ids_to_label = self.unlabeled_datasets[0].pick_random_minority_instances(k)
        majority_ids_to_label = self.unlabeled_datasets[0].pick_random_majority_instances(k)
        all_ids_to_label = [inst.id for inst in minority_ids_to_label + majority_ids_to_label]
        return all_ids_to_label
        
        
    def get_labeled_instance_ids(self):
        return self.labeled_datasets[0].labeled_instances()
        
    def undersample_labeled_datasets(self, k=None):
        '''
        Returns undersampled copies of the current labeled datasets, i.e., copies in which
        the two classes have equal size. Note that this methods returns a *copy* of the 
        undersampled datasets. Thus it *does not mutate the labeled datasets*.
        '''
        if self.labeled_datasets and len(self.labeled_datasets) and (len(self.labeled_datasets[0].instances) > 0):
            if k is None:
                print "undersampling majority class to equal that of the minority examples"
                k = self.labeled_datasets[0].number_of_majority_examples() - self.labeled_datasets[0].number_of_minority_examples()
            # we copy the datasets rather than mutate the class members.
            copied_datasets = [d.copy() for d in self.labeled_datasets]
            if k < self.labeled_datasets[0].number_of_majority_examples() and k > 0:
                # make sure we have enough majority examples...
                print "removing %s majority instances. there are %s total majority examples in the dataset." % \
                        (k, self.labeled_datasets[0].number_of_majority_examples())
                removed_instance_ids = copied_datasets[0].undersample(k)
                # if there is more than one feature-space, remove the same 
                # instances from the remaining spaces (sets)
                for labeled_dataset in copied_datasets[1:]:
                    # now remove them from the corresponding sets
                    labeled_dataset.remove_instances(removed_instance_ids)
        else:
            raise Exception, "No labeled data has been provided!"   
        return copied_datasets

    def get_random_unlabeled_ids(self, k):
        ''' Returns a random set of k instance ids ''' 
        return random.sample(self.unlabeled_datasets[0].get_instance_ids(), k)

    def rebuild_models(self, undersample_first=False):
        raise Exception, "No models provided! (BaseLearner)"

    def write_out_labeled_data(self, path, dindex=0):
        outf = open(path, 'w')
        outf.write(self.labeled_datasets[dindex].get_points_str())
        outf.close()

    def unlabel_instances(self, inst_ids):
        for inst_index in range(len(self.labeled_datasets[0].instances)):
            if self.labeled_datasets[0].instances[inst_index].id in inst_ids:
                for unlabeled_dataset, labeled_dataset in zip(self.unlabeled_datasets, self.labeled_datasets):
                    labeled_dataset.instances[inst_index].lbl = labeled_dataset.instances[inst_index].label
                    labeled_dataset.instances[inst_index].has_synthetic_label = False

        # now remove the instances and place them into the unlabeled set
        for unlabeled_dataset, labeled_dataset in zip(self.unlabeled_datasets, self.labeled_datasets):
            unlabeled_dataset.add_instances(labeled_dataset.remove_instances(inst_ids))

def _arg_max(ls, f):
    ''' Returns the index for x in ls for which f(x) is maximal w.r.t. the rest of the list '''
    return_index = 0
    max_val = f(ls[0])
    for i in range(len(ls)-1):
        if f(ls[i+1]) > max_val:
            return_index = i
            max_val = f(ls[i+1])
    return return_index

        