'''    
    Byron C Wallace
    Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
        
    base_learner.py
    --
    This module contains a base learner class, which you can subclass  to implement your own 
    active learning strategy. 
'''

import pdb
import random
import math
import svm
from svm import *
import dataset 
import cluster
import numpy
import smote


def evaluate_learner_with_holdout(learner, test_set):
    '''
    If you're not considering a "finite pool" problem, this is really the correct way to evaluate the trained classifiers. 
    
    @params
    learner -- the learner to be evaluated
    ''''
    results={}
    pos_count = learner.labeled_datasets[0].number_of_minority_examples()
    neg_count = learner.labeled_datasets[0].number_of_majority_examples()
    print "positives found during learning: %s\nnegatives found during learning: %s" % (pos_count, neg_count)
    print "evaluating learner over %s instances." % len(learner.unlabeled_datasets[0].instances)
    fns = 0
    predictions = []
    point_sets = [dataset.get_samples() for dataset in test_sets]
    # the labels are assumed to be the same; thus we only use the labels for the first dataset
    true_labels = test_sets[0].get_labels()
   
    # loop over all of the examples, and feed to the "cautious_classify" method 
    # the corresponding point in each feature-space
    for example_index in range(len(point_sets[0])):
        # hand the cautious_predict method a list of representations of x; one per feature space/model
        prediction = learner.predict([point_sets[feature_space_index][example_index] for feature_space_index in range(len(point_sets))])
        predictions.append(prediction)
    
    conf_mat =  svm.evaluate_predictions(predictions, true_labels)
    print "confusion matrix:"
    print conf_mat
    results["npos"] = pos_count
    results["confusion_matrix"] = conf_mat
    results["accuracy"] = float (conf_mat["tp"] + conf_mat["tn"]) / float(sum([conf_mat[key] for key in conf_mat.keys()]))
    if float(conf_mat["tp"]) == 0:
        results["sensitivity"] = 0
    else:
        results["sensitivity"] = float(conf_mat["tp"]) / float(conf_mat["tp"] + conf_mat["fn"])
    print "sensitivity: %s" % results["sensitivity"]
    print "accuracy: %s" % results["accuracy"]
    return results
    
def evaluate_learner(learner, include_labeled_data_in_metrics=True):
    '''
    Returns a dictionary containing various metrics for learner performance, as measured over the
    examples in the unlabeled_datasets belonging to the learner.
    
    @parameters
    include_labeled_data_in_metrics -- If this is true, the (labeled) examples in the learner's labeled_datasets field
                                                                                will be included in evaluation. Useful for 'finite' pool learniner; otherwise misleading.
                                                                                In general, one should use a holdout.
    '''
    tps, tns, fps = 0,0,0
    results = {}
    # first we count the number of true positives and true negatives discovered in learning. this is so we do not
    # unfairly penalize active learning strategies for finding lots of the minority class during training.
    if include_labeled_data_in_metrics:
        tps = learner.labeled_datasets[0].number_of_minority_examples(use_real_label=True, include_synthetics=False)
        tns = learner.labeled_datasets[0].number_of_majority_examples()
        fps = learner.labeled_datasets[0].number_of_false_minorities()
    results["npos"] = tps
    
    print "positives found during learning: %s\nnegatives found during learning: %s" % (tps, tns)
    print "number of *synthetics* used in training: %s" % len(learner.labeled_datasets[0].get_synthetic_ids())
    print "evaluating learner over %s instances." % len(learner.unlabeled_datasets[0].instances)
    fns = 0
    predictions = []

    # get the raw points out for prediction
    point_sets = [dataset.get_samples() for dataset in learner.unlabeled_datasets]
    # the labels are assumed to be the same; thus we only use the labels for the first dataset
    true_labels = learner.unlabeled_datasets[0].get_labels()
    # loop over all of the examples, and feed to the "cautious_classify" method 
    # the corresponding point in each feature-space
    for example_index in range(len(point_sets[0])):
        # hand the cautious_predict method a list of representations of x; one per feature space/model
        prediction = learner.predict([point_sets[feature_space_index][example_index] for feature_space_index in range(len(point_sets))])
        predictions.append(prediction)

        
    conf_mat =  svm.evaluate_predictions(predictions, true_labels)
    # 
    # evaluate_predictions does not include the instances found during training!
    #
    conf_mat["tp"]+= tps
    conf_mat["tn"]+= tns
    conf_mat["fp"]+= fps
    print "confusion matrix:"
    print conf_mat
    results["confusion_matrix"] = conf_mat
    results["accuracy"] = float (conf_mat["tp"] + conf_mat["tn"]) / float(sum([conf_mat[key] for key in conf_mat.keys()]))
    if float(conf_mat["tp"]) == 0:
        results["sensitivity"] = 0
    else:
        results["sensitivity"] = float(conf_mat["tp"]) / float(conf_mat["tp"] + conf_mat["fn"])
    print "sensitivity: %s" % results["sensitivity"]
    print "accuracy: %s" % results["accuracy"]
    return results
    
class BaseLearner:
    
    def __init__(self, unlabeled_datasets = [], models=None):
        # just using default parameter for now
        
        # by default, vanilla svm_parameter object is used. overwrite if you want.
        print "using default svm parameters!"
        self.params = [svm_parameter()  for d in unlabeled_datasets]
        self.unlabeled_datasets = unlabeled_datasets
        # initialize empty labeled datasets (i.e., all data is unlabeled to begin with)
        self.labeled_datasets = [dataset.dataset([]) for d in unlabeled_datasets]
        self.models = models
        self.div_hash = {}
        self.dist_hash = {}
        self.k_hash = {}
        self.euclid_hash = {}
        self.clustering = None
        self.explore_mode = False
        self.faux_minorities = []
        self.hypersmote = False
        self.iter = 0
        
        # default prediction function; only important if you're aggregating multiple feature spaces (see 
        # cautious_predict function documentation)
        self.predict = self.cautious_predict
 
        
    def active_learn(self, num_examples_to_label, query_function = None, num_to_label_at_each_iteration=10, 
                                            rebuild_models_at_each_iter=True):
        ''''
        Core active learning loop. Uses the provided query function (query_function) to select a number of examples 
        (num_to_label_at_each_iteration) to label at each step, until the total number of examples requested 
        (num_examples_to_label) has been labeled. The models will be updated at each iteration.
        '''
        if not query_function:
            query_function = self.SIMPLE
        
        
        labeled_so_far = 0
        while labeled_so_far < num_examples_to_label:
            
            if self.is_osugi:
                query_function = self.osugi_explore()
            
            print "labeled %s out of %s" % (labeled_so_far, num_examples_to_label)
            example_ids_to_label = query_function(num_to_label_at_each_iteration)
            # now remove the selected examples from the unlabeled sets and put them in the labeled sets.
            # if not ids are returned -- ie., if a void query_function is used --
            # it is assumed the query function took care of labeling the examples selected. See, e.g.,
            # the maximally_diverse_method
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

       
    def write_out_minorities(self, path, dindex=0):
        outf = open(path, 'w')
        outf.write(self.labeled_datasets[dindex].get_minority_examples().get_points_str())
        outf.close()
        
    def write_out_labeled_data(self, path, dindex=0):
        outf = open(path, 'w')
        outf.write(self.labeled_datasets[dindex].get_points_str())
        outf.close()
        
        
    def unlabel_instances(self, inst_ids):
        for inst_index in range(len(self.labeled_datasets[0].instances)):
            if self.labeled_datasets[0].instances[inst_index].id in inst_ids:
                for unlabeled_dataset, labeled_dataset in zip(self.unlabeled_datasets, self.labeled_datasets):
                    labeled_dataset.instances[inst_index].lbl = labeled_dataset.instances[inst_index].real_label
                    labeled_dataset.instances[inst_index].has_synthetic_label = False
        
        # now remove the instances and place them into the unlabeled set
        for unlabeled_dataset, labeled_dataset in zip(self.unlabeled_datasets, self.labeled_datasets):
            unlabeled_dataset.add_instances(labeled_dataset.remove_instances(inst_ids))

                         
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
    
    def cautious_predict(self, X):
        '''
        A naive way of combining different models (built over different feature-spaces); if any othe models vote yes, then vote yes.
        When there is only on feature space, this reduces to simply "predict".
        ''''
        if self.models and len(self.models):
            return max([m.predict(x) for m,x in zip(self.models, X)])
        else:
            raise Exception, "No models have been initialized."
        
    
    def pick_balanced_initial_training_set(self, k):
        '''
        Picks k + and k - examples at random for bootstrap set.
        '''
        minority_ids_to_label = self.unlabeled_datasets[0].pick_random_minority_instances(k)
        majority_ids_to_label = self.unlabeled_datasets[0].pick_random_majority_instances(k)
        all_ids_to_label = minority_ids_to_label + majority_ids_to_label
        self.label_instances_in_all_datasets(all_ids_to_label)
        return all_ids_to_label
        
        
    def pick_initial_training_set(self, k, build_models=True):
        '''
        Select a set of training examples from the dataset(s) at random. This set will be used
        to build the initial model. The **same training examples will be selected from each dataset.
        '''
        self.label_at_random(k)
        if build_models:
            print "building models..."
            self.rebuild_models()
            print "done."
    
        
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
    
 
    def label_at_random(self, k):
        '''
        Select and 'label' a set of k examples from the (unlabeled) dataset(s) at random. 
        '''
        if self.unlabeled_datasets and len(self.unlabeled_datasets):
            # remove a random subset of instances from one of our datasets (it doesn't matter which one)
            removed_instances = self.unlabeled_datasets[0].get_and_remove_random_subset(k)
            # add this set to the labeled data
            self.labeled_datasets[0].add_instances(removed_instances)
            # get the removed instance numbers
            removed_instance_nums = [inst.id for inst in removed_instances]
            # if there is more than one feature-space, remove the same instances from the remaining spaces (sets)
            for unlabeled_dataset, labeled_dataset in zip(self.unlabeled_datasets[1:], self.labeled_datasets[1:]):
                # now remove them from the corresponding sets
                labeled_dataset.add_instances(unlabeled_dataset.remove_instances(removed_instance_nums))
        else:
            raise Exception, "No datasets have been provided!"
        
        
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
        

    def rebuild_models(self, undersample_first=False, undersample_cleverly=False, include_synthetics=True):
        '''
        Rebuilds all models over the current labeled datasets.
        '''    
        if not include_synthetics:
            print "removing synthetics from training data!"
            datasets = self.get_non_synthetics()
        else:
            print "including synthetics in training data!"
            
        if undersample_first:
            print "undersampling before building models.."
            if undersample_cleverly:
                print "undersampling cleverly!"
                datasets = self.undersample_labeled_datasets_cleverly()
            else:
                datasets = self.undersample_labeled_datasets()
                print "done."
        else:
            datasets = self.labeled_datasets
            
        print "training model(s) on %s instances" % len(datasets[0].instances)
        self.models = []
        for dataset, param in zip(datasets, self.params):
            samples, labels = dataset.get_samples_and_labels(include_synthetics=include_synthetics)
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
    



        