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

class BaseLearner:
    '''
    Base learner class. Sub-class this object to implement your own learning strategy. 
    
    Repeating the comment in curious_snake.py, Curious Snake was originally written for a scenario in which multiple feature spaces
    were being exploited, thus pluralizing may of the attributes in this class. For example, 
    *lists* of unlabeled_datasets and models are kept. If you only have one feature space that you're interested
     in, as is often the case, simply pass around unary lists.  
    ''' 
    
    # TODO add (optional?) schohn/general stopping criterion implementation  -- where should this go?
    
    def __init__(self, unlabeled_datasets = [], models = None):
        '''
        unlabeled_datasets should be either (1) a string pointing to a single data file (e.g., "mydata.txt") or (2) a list of strings
        pointing to multiple data files that represent the same data with different feature spaces. For more on the data format,
        consult the doc or see the samples.
        '''
        if type(unlabeled_datasets) == type(""):
            # then a string, presumably pointing to a single data file, was passed in
            unlabeled_datasets  = [unlabeled_datasets]
            
        self.unlabeled_datasets = unlabeled_datasets
        # initialize empty labeled datasets (i.e., all data is unlabeled to begin with)
        self.labeled_datasets = [dataset.dataset([]) for d in unlabeled_datasets]
        self.models = models

        self.query_function = self.base_q_function # throws exception if not overridden 
        self.name = "Base"
        
        # default prediction function; only important if you're aggregating multiple feature spaces (see 
        # cautious_predict function documentation)
        self.predict = self.majority_predict
 
    
    def active_learn(self, num_examples_to_label, num_to_label_at_each_iteration=5, 
                                                rebuild_models_at_each_iter=True):
        ''''
        Core active learning routine. Here the learner uses its query function to select a number of examples 
        (num_to_label_at_each_iteration) to label at each step, until the total number of examples requested 
        (num_examples_to_label) has been labeled. The models will be updated at each iteration.
        '''
        labeled_so_far = 0
        while labeled_so_far < num_examples_to_label:
            print "labeled %s out of %s" % (labeled_so_far, num_examples_to_label)
            example_ids_to_label = self.query_function(num_to_label_at_each_iteration)
            # now remove the selected examples from the unlabeled sets and put them in the labeled sets.
            # if not ids are returned -- ie., if a void query_function is used --
            # it is assumed the query function took care of labeling the examples selected. 
            if example_ids_to_label:
                self.label_instances_in_all_datasets(example_ids_to_label)

            if rebuild_models_at_each_iter:
                self.rebuild_models()
                print "models rebuilt with %s labeled examples" % len(self.labeled_datasets[0].instances)    
            else:
                print "model has %s labeled examples thus far (not rebuilding models @ each iter)" % len(self.labeled_datasets[0].instances)

            labeled_so_far += num_to_label_at_each_iteration

        self.rebuild_models()
        print "active learning loop completed; models rebuilt."
            
    
    def base_q_function(self, k):
        ''' overwite this method with query function of choice (e.g., SIMPLE) '''
        raise Exception, "no query function provided!"

                                 
    def label_all_data(self):
        '''
        Labels all the examples in the training set
        '''
        inst_ids = [inst.id for inst in self.unlabeled_datasets[0].instances]
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
        all_ids_to_label = minority_ids_to_label + majority_ids_to_label
        self.label_instances_in_all_datasets(all_ids_to_label)
        return all_ids_to_label
        
        
    def undersample_labeled_datasets(self, k=None):
        '''
        Undersamples the current labeled datasets, i.e., makes the two classes of equal sizes. 
        Note that this methods returns a *copy* of the undersampled datasets. Thus it
        *does not mutate the labeled datasets*.
        '''
        if self.labeled_datasets and len(self.labeled_datasets) and (len(self.labeled_datasets[0].instances)):
            if not k:
                print "undersampling majority class to equal that of the minority examples"
                # we have to include 'false' minorities -- i.e., instances we've assumed are positives -- because otherwise we'd be cheating
                k = self.labeled_datasets[0].number_of_majority_examples() - self.labeled_datasets[0].number_of_minority_examples()
            # we copy the datasets rather than mutate the class members.
            copied_datasets = [dataset.dataset(list(d.instances)) for d in self.labeled_datasets]
            if k < self.labeled_datasets[0].number_of_majority_examples() and k > 0:
                # make sure we have enough majority examples...
                print "removing %s majority instances. there are %s total majority examples in the dataset." % (k, self.labeled_datasets[0].number_of_majority_examples())
                removed_instances = copied_datasets[0].undersample(k)
                # get the removed instance numbers
                removed_instance_nums = [inst.id for inst in removed_instances]
                # if there is more than one feature-space, remove the same instances from the remaining spaces (sets)
                for labeled_dataset in copied_datasets[1:]:
                    # now remove them from the corresponding sets
                    labeled_dataset.remove_instances(removed_instance_nums)
        else:
            raise Exception, "No labeled data has been provided!"   
        return copied_datasets
    
         
    def get_random_unlabeled_ids(self, k):
        '''
        Returns a random set of k instance ids
        ''' 
        selected_ids = []
        ids = self.unlabeled_datasets[0].get_instance_ids()  
        for i in range(k):
            random_id = random.choice(ids)
            ids.remove(random_id)
            selected_ids.append(random_id)
        return selected_ids
        

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

        