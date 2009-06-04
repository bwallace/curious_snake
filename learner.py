'''
###############################################

    learner.py
    Byron C Wallace
    Tufts Medical Center
        
    This module represents a learner. Includes active learning. 

###############################################
'''

import pdb
import random
import math
import svm
from svm import *
import dataset 
import cluster
import KMeans
import numpy
import Stats
import smote

def evaluate_learner_with_holdout(learner, test_sets):
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
        prediction = learner.cautious_predict([point_sets[feature_space_index][example_index] for feature_space_index in range(len(point_sets))])
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
    
def evaluate_learner(learner, include_labeled_data_in_metrics=True, stacked=False):
    '''
    Returns a dictionary containing various metrics for learner performance, as measured over the
    examples in the unlabeled_datasets belonging to the learner.
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
    if not stacked:
        # get the raw points out for prediction
        point_sets = [dataset.get_samples() for dataset in learner.unlabeled_datasets]
        # the labels are assumed to be the same; thus we only use the labels for the first dataset
        true_labels = learner.unlabeled_datasets[0].get_labels()
        # loop over all of the examples, and feed to the "cautious_classify" method 
        # the corresponding point in each feature-space
        for example_index in range(len(point_sets[0])):
            # hand the cautious_predict method a list of representations of x; one per feature space/model
            prediction = learner.cautious_predict([point_sets[feature_space_index][example_index] for feature_space_index in range(len(point_sets))])
            predictions.append(prediction)
    else:
        meta_model = learner._get_stacked_meta_model(undersample_first=True)
        unlabeled = learner._meta_dataset(learner.unlabeled_datasets)
        point_set = unlabeled.get_samples()
        true_labels = unlabeled.get_labels()
        for example in point_set:
            prediction = meta_model.predict(example)
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
    
class learner:
    labeled_datasets = []
    unlabeled_datasets = []
    picked_during_al = []
    # we need a param for each model
    params = []
    models = None
    
    # for osugi
    explored_last_q = False
    
    def __init__(self, unlabeled_datasets = [], models=None):
        # just using default parameter for now
        self.params = [svm_parameter(weight=[1, 1000], kernel_type=LINEAR)  for d in unlabeled_datasets]
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
        self.lam = 40
        
        # osugi
        self.is_osugi = False
        self.X_for_osugi = None
        self.osugi_p = .5
        self.H = None
        self.H_prime = None
        self.explored_last_q = False
        self.num_explores = 0
        self.num_exploits = 0
        
    def active_learn(self, num_examples_to_label, query_function = None, num_to_label_at_each_iteration=10, 
                            rebuild_models_at_each_iter=True, lbl_closest_as_positive = False):
        ''''
        Active learning loop. Uses the provided query function (query_function) to select a number of examples 
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
                if self.is_osugi and self.explored_last_q:
                    print "updating H !"
                    self.H_prime = [self.models[0].predict(x) for x in self.X_for_osugi]
                    if not self.H:
                        self.H = self.H_prime
                    else:
                        s1 = numpy.dot(self.H, self.H_prime) / ((math.sqrt(numpy.dot(self.H, self.H))) * (math.sqrt(numpy.dot(self.H_prime, self.H_prime))))
                        d1 = 3 - 4*s1
                        print "d1: %s " % d1
                        _lambda = .1
                        eps = .01
                        self.osugi_p = max(min(self.osugi_p * _lambda * math.exp(d1), 1 - eps), eps)
                        self.H = self.H_prime
    
            else:
                print "model has %s labeled examples thus far (not rebuilding models @ each iter)" % len(self.labeled_datasets[0].instances)
            
            if lbl_closest_as_positive:
                closest_ids = self.SIMPLE(num_to_label_at_each_iteration)
                # 'label' the closest instances as being in class '1', even if they're not
                # this is 'free' except we might be introducing (one-way) error
                self.set_labels_for_instances(closest_ids)
                self.label_instances_in_all_datasets(closest_ids)
            labeled_so_far += num_to_label_at_each_iteration
        
        #if self.hypersmote:
        #    self.hyper_smote()
            
        self.rebuild_models()
        print "active learning loop completed; models rebuilt."

    def cluster_unlabeled_data(self, use_subset=True):
        #subset_size = 500
        points = self.unlabeled_datasets[0].instances
        print "using subset!"
        #f = lambda x,y: self._compute_cos(self.models[0], x, y)
        f = lambda x,y,: self.euclid_dist(x, y)
        self.clustering = cluster.HierarchicalClustering(points, f)
        
        
    def write_out_minorities(self, path):
        outf = open(path, 'w')
        outf.write(self.labeled_datasets[0].get_minority_examples().get_points_str())
        outf.close()
        
    def write_out_labeled_data(self, path):
        outf = open(path, 'w')
        outf.write(self.labeled_datasets[0].get_points_str())
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
                    
    def set_labels_for_instances(self, inst_ids, lbl=1.0):
        '''
        Regardless of the 'true' label of the instances, this method labels them as '1.0'
        '''
        for inst_index in range(len(self.unlabeled_datasets[0].instances)):
            if self.unlabeled_datasets[0].instances[inst_index].id in inst_ids:
                for ds in self.unlabeled_datasets:
                    ds.instances[inst_index].set_synthetic_label(lbl)
                    
     
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
            
    def get_non_synthetics(self):
        copied_datasets = [dataset.dataset(list(d.instances)) for d in self.labeled_datasets]
        synth_ids = copied_datasets[0].get_synthetic_ids()
        for labeled_dataset in copied_datasets:
            # now remove them from the corresponding sets
            labeled_dataset.remove_instances(synth_ids)
        return copied_datasets
            
    def undersample_labeled_datasets_cleverly(self, k=None):
        '''
        Undersamples the current labeled datasets; returns a *copy* of the undersampled datasets.
        *Does not mutate the labeled datasets*.
        '''
        if self.labeled_datasets and len(self.labeled_datasets) and (len(self.labeled_datasets[0].instances)):
            if not k:
                print "undersampling majority class to equal that of the minority examples"
                # we have to include 'false' minorities -- i.e., instances we've assumed are positives -- because otherwise we'd be cheating
                k = self.labeled_datasets[0].number_of_majority_examples() - self.labeled_datasets[0].number_of_minority_examples(include_synthetics=True)
            # we copy the datasets rather than mutate the class members.
            copied_datasets = [dataset.dataset(list(d.instances)) for d in self.labeled_datasets]
            if k < self.labeled_datasets[0].number_of_majority_examples() and k > 0:
                # make sure we have enough majority examples...
                print "removing %s majority instances. there are %s total majority examples in the dataset." % (k, self.labeled_datasets[0].number_of_majority_examples())
                
                # remove closeset to the hyperplane
                print k
                removed_these = [x_id for x_id in self._SIMPLE(self.models[0], self.labeled_datasets[0].get_majority_examples(), int(1*k))]
                # if there is more than one feature-space, remove the same instances from the remaining spaces (sets)
                for labeled_dataset in copied_datasets:
                    # now remove them from the corresponding sets
                    labeled_dataset.remove_instances(removed_these)
                
                # now remove the rest
                '''
                removed_instances = copied_datasets[0].undersample(int(0 *k))
                removed_instance_nums = [inst.id for inst in removed_instances]
                # if there is more than one feature-space, remove the same instances from the remaining spaces (sets)
                for labeled_dataset in copied_datasets[1:]:
                    # now remove them from the corresponding sets
                    labeled_dataset.remove_instances(removed_instance_nums)
                '''
                
        else:
            raise Exception, "No labeled data has been provided!"   
        return copied_datasets
            
        
    def undersample_labeled_datasets(self, k=None):
        '''
        Undersamples the current labeled datasets; returns a *copy* of the undersampled datasets.
        *Does not mutate the labeled datasets*.
        '''
        if self.labeled_datasets and len(self.labeled_datasets) and (len(self.labeled_datasets[0].instances)):
            if not k:
                print "undersampling majority class to equal that of the minority examples"
                # we have to include 'false' minorities -- i.e., instances we've assumed are positives -- because otherwise we'd be cheating
                k = self.labeled_datasets[0].number_of_majority_examples() - self.labeled_datasets[0].number_of_minority_examples(include_synthetics=True)
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
    
    def osugi_explore(self):
        if random.random() <= self.osugi_p:
            # explore with probability p
            self.explored_last_q = True
            print "EXPLORING (kff)"
            return self.kff
        else:
            print "EXPLOITING"
            self.explored_last_q = False
            return self.SIMPLE
            
    def hyper_dist_SIMPLE_hybrid(self, k):
        ids_to_lbl = None
        self.explore_mode = False
        self.iter +=1 
        print "iteration: %s" % self.iter
        if self.explore_mode:
            print "\n\nDISTS"
            ids_to_lbl = self.hyper_dist(k)
            self.label_instances_in_all_datasets(ids_to_lbl)
            intersection = [cur_id for cur_id in ids_to_lbl if cur_id in self.unlabeled_datasets[0].get_list_of_minority_ids()]
            if len(intersection):
                print "\n\nFOUND POSITIVE.. rebuilding models"
                #pdb.set_trace()
                self.explore_mode= False
                self.rebuild_models()
        else:
            print "\n\nUSING SIMPLE!"
            ids_to_lbl = self.SIMPLE(k)
            self.explore_mode= True
            
            self.label_instances_in_all_datasets(ids_to_lbl)
            
            print "ran simple ... now introducing fake labels.. (HYPER SMOTE)"
            #self.lam = max(self.lam -1, 0)
            #print "subtracting %s from lam; rate is now" % self.lam
            #nsynths = k
            #if k % 2 == 0:
            #    nsynths = k/2
            
            nsynths = self.labeled_datasets[0].number_of_minority_examples(include_synthetics=False)
            self.hyper_smote(alpha = self.lam, nsynths=nsynths)
            print "removing examples more than 10 iterations old"
                
            print "done. rebuilding models.."
            self.rebuild_models()
        #simple_ids = self.SIMPLE(k/2)
        
        #self.rebuild_models()
        #hyper_ids = self.hyper_dist(k/2)
        #return hyper_ids
        
    def hyper_smote(self, nclusters=10, alpha=20, nsynths=10, ppos=.5):
        model = self.models[0]
        data = self.unlabeled_datasets[0]
        labeled_data = self.labeled_datasets[0]
        id_dist_pairs = []
        points = []
        point_vals = []

        for inst in data.instances:
            point = model.distance_to_hyperplane(inst.point, signed=True)
            point_vals.append(point)
            points.append(KMeans.Point([point], reference=inst))
        
        clustering = KMeans.kmeans(points, nclusters, .00001)
        best_clust = self.find_good_cluster(clustering)
        #best_clust = self.find_good_cluster(clustering)
        if not best_clust:
            print "\n\nNO CLUSTERS FOUND\n\n"
            return None
        print "ALPHA = NSYNTHS!!!"
        alpha = nsynths
        #points_in_best_cluster = [p.reference for p in best_clust.points]
        
        cur_n = len([p for p in self.labeled_datasets[0].instances if not p.has_synthetic_label])
        if cur_n < 400:
            print "hyper SMOTEING"
            if self.iter < 200:
                print "not including the minorities we've already discovered!"
            minorities = self.labeled_datasets[0].get_minority_examples()
            for i in range(nsynths):
                best_clust = self.random_neg_cluster(clustering)
                points_in_best_cluster = [p.reference for p in best_clust.points]
                if self.iter < 50:
                    points_in_best_cluster.extend(minorities.instances)
                synthetics = smote.SMOTE_n_points(points_in_best_cluster, 1)
                #print "done"
                # now we set the ids to ensure uniqueness. synthetic ids will be 
                # given the id -1 * n where n is the number of synthetics add thus far minus one
                # (i.e., if this is the nth synthetic added, the id will be -n)
                num_synths = len(data.get_synthetic_ids())
                for synth in synthetics:
                    num_synths = num_synths+1
                    synth.id = -1*num_synths
            
                self.faux_minorities.extend(synthetics)
                labeled_data.add_instances(synthetics)
        
            print "faux min.: %s, labeled data: %s" % (len(self.faux_minorities), len(labeled_data.instances))
        else:
            if len(self.faux_minorities):
                "\n\n No MORE SYNTHETICS -- REMOVING THE LAST OF THEM!"
                remove_these = self.faux_minorities
                self.faux_minorities = []
                labeled_data.remove_instances([inst.id for inst in remove_these])
                print "done."
            
        if len(self.faux_minorities) > alpha:
            num_to_remove = len(self.faux_minorities) - alpha
            print "removing an expired fauxample." 
            remove_these_synthetics = self.faux_minorities[:num_to_remove]
            self.faux_minorities = self.faux_minorities[num_to_remove:]
            labeled_data.remove_instances([inst.id for inst in remove_these_synthetics])
            
            #self.unlabel_instances([inst.id for inst in unlabel_these])
            #self.faux_minorities = self.faux_minorities[:alpha]
            print "amount of labeled data after removing instance: %s" % len(self.labeled_datasets[0].instances)
        '''
        neg_clusters = [c for c in clustering if c.centroid.coords[0] < 0]
        fakepoints = []
        for clust in neg_clusters:
            point_closest_to_centroid = self.closest_to_centroid(clust, points, point_vals)
            fakepoints.append(point_closest_to_centroid)
        '''
        # fake it
        #pdb.set_trace()
        #self.set_labels_for_instances([fp.id for fp in synthetics])
       # pdb.set_trace()
       
        #self.label_instances_in_all_datasets([fp.id for fp in fakepoints])
        #self.unlabel_instances([fp.id for fp in fakepoints])
       
        
    def closest_to_centroid(self, clust, point_list, point_vals):
        closest_p, closest_dist = point_list[0].reference, abs(point_vals[0] - clust.centroid.coords[0])
        for point, pval in zip(point_list[1:], point_vals[1:]):
            cur_dist = abs(pval - clust.centroid.coords[0])
            if cur_dist < closest_dist:
                closest_p = point.reference
                closest_dist = cur_dist
        return closest_p
        
    def hyper_dist(self, k):
        model = self.models[0]
        #data = dataset.dataset(self.unlabeled_datasets[0].get_random_subset(500))
        data = self.unlabeled_datasets[0]
        id_dist_pairs = []
        points = []
        point_vals = []
        for inst in data.instances:
            #all_distances[inst.id] = model.distance_to_hyperplane(inst.point)
            #id_dist_pairs.append((inst, model.distance_to_hyperplane(inst.point, signed=True)))
            point = model.distance_to_hyperplane(inst.point, signed=True)
            point_vals.append(point)
            points.append(KMeans.Point([point], reference=inst))
        
        #num_clusts = len(data.instances)/25 
        #print num_clusts
        #pdb.set_trace()
        num_clusts = len(data.instances)/30 
        clustering = KMeans.kmeans(points, num_clusts, .00001)
        #pdb.set_trace()
       # self.A(clustering[0])
        
        #cutoff = 1
        #while (len(clustering)) > 10:
        #    clustering = self.G_means(clustering[0], cutoff)
        #    cutoff = cutoff+50
        #f = lambda x,y: abs(x[1] - y[1])
        #clustering = cluster.HierarchicalClustering(id_dist_pairs, f)
        #clusters = cluster.KMeansClustering(20)
        #reclustering = KMeans.kmeans(c.points, 2, .0001)
        # compute average difference in distance to hyperplane between labeled positive examples 
        #pos_distances = [model.distance_to_hyperplane(x.point, signed=True) for x in self.labeled_datasets[0].get_minority_examples().instances]
        #all_distances = [model.distance_to_hyperplane(x.point, signed=True) for x in self.labeled_datasets[0].instances]
        #level = 1.0
        #pdb.set_trace()
        #clusters = clustering.getlevel(level)
        #pdb.set_trace()
        #beta = 300

        #while (len(clusters) < 10):
        #while (max(clust_size) >= beta*min(clust_size) or len(clust_size) == 1):
       # while (len(clusters) < 20):
    #        clusters = clustering.getlevel(level)
    #        clust_size = [len(c) for c in clusters]
    #        level-=.00005
            
        #all_pairs = [(x,y) for i,x in enumerate(all_distances) for y in all_distances[i+1:]]
        #sum_diffs = sum([abs(pair[0] - pair[1]) for pair in all_pairs])
        #avg_diff = sum_diffs / len(all_pairs)
        
        lbl_these_ids = []
        while len(lbl_these_ids) < k:
            num_left_to_label = k - len(lbl_these_ids)
            clust = self.find_good_cluster(clustering)
            actual_lbls = [p.reference.label for p in clust.points if p.reference.label > 0]
            print "size of selected cluster: %s:" % len(clust.points)
            
            print "num positives in cluster: %s" % len(actual_lbls)
            #pdb.set_trace()
            clustering.remove(clust)
            inst_ids = [p.reference.id for p in clust.points]
            if len(inst_ids) <= num_left_to_label:
                lbl_these_ids.extend(inst_ids)
            else:
                lbl_these_ids.extend(random.sample(inst_ids, num_left_to_label))
            
        return lbl_these_ids
    
   # def _reasonable_clustering(clustering):
#         '''
#        Finds and returns a 'reasonble' clustering using a greedy strategy
#        '''
    
    def random_neg_cluster(self, clusters):
        neg_clusters = [c for c in clusters if c.centroid.coords[0] < 0]
        return random.choice(neg_clusters)
        
    def find_good_cluster(self, clusters):
        # exclude positive clusters; we're already going to label these as +1
        neg_clusters = [c for c in clusters if c.centroid.coords[0] < 0]
        if not len(neg_clusters):
            print "no clusters have a negative centroid. returning None."
            return None
            
        distances_to_h = [abs(c.centroid.coords[0]) for c in neg_clusters]
        max_dist = max(distances_to_h)
        points = [[p.coords[0] for p in c.points] for c in clusters]
        std_devs = [Stats.Statistics(p).stddev for p in points]
        max_std_div= max(std_devs)
        sizes = [len(c.points) for c in neg_clusters]
        max_size = max(sizes)
        score_cluster = lambda i: distances_to_h[i]/max_dist + std_devs[i]/max_std_div
        best_cluster, best_cluster_score = neg_clusters[0], score_cluster(0)
        for c_index in range(len(neg_clusters)-1):
            i = c_index+1
            if score_cluster(i) > best_cluster_score:
                best_cluster = neg_clusters[i]
                best_cluster_score = score_cluster(i)
        return best_cluster
        
        
        
        
    def G_means(self, c, alpha):
        if self.A(c) <= alpha:
            return [c]
        reclustering = KMeans.kmeans(c.points, 2, .0001)
        #pdb.set_trace()
        return self.G_means(reclustering[0], alpha) + self.G_means(reclustering[1], alpha)
        
    def A(self, c):
        '''
        c is a cluster
        '''
        if len(c.points) <= 1:
            return 0
            
        # again, assuming scalar values
        point_vals = [x.coords[0] for x in c.points]
        val_stats = Stats.Statistics(point_vals) # assumes point_vals is a list of scalars!
        adjusted_point_vals = [ (x - val_stats.mean) / val_stats.stddev for x in point_vals ]
        F = lambda x: 1 / sqrt(2*math.pi) * math.exp(-1 * math.pow(x, 2)/2)
        #pdb.set_trace()
        adjusted_point_vals.sort()
        A_stat = 0
        n = len(adjusted_point_vals)
        for point_index in range(n):
            i = point_index+1
            z_i = F(adjusted_point_vals[point_index])
            #pdb.set_trace()
            A_stat += (2*i-1) * (math.log(z_i) + math.log(1-F(adjusted_point_vals[n-i]))) - n
        A_stat = -1.0/n * A_stat
        A_star = A_stat * (1 + 4.0/n - 25.0/(math.pow(n,2)))
        return A_star
        
        
    def _adjust_vals(self, values):
        '''
        Returns a new list of values s.t. that mean has been adjusted to 0 and the std. dev. to 1.
        '''
        val_stats = Stats.Statistics(point_vals)
        return [ (x - val_stats.mean) / val_stats.stddev for x in values]
        
    def _clustering_score(self, list_of_clusters):
        n_c = len(list_of_clusters) # number of clusters
        N = sum([len(c) for c in list_of_clusters]) # total number of examples
        clust_score = n_c / float(N) + self._distribution_score(list_of_clusters)
        return clust_score
        
    def _distribution_score(self, list_of_clusters):
        N = sum([len(c) for c in list_of_clusters])
        avg_cluster_size = N / float(len(list_of_clusters))
        avg_score = avg_cluster_size / float(N)
        avg_variance = sum([abs(len(c) - avg_cluster_size) for c in list_of_clusters])/float(len(list_of_clusters))
        var_score = 1.0
        if variance:
            var_score = 1.0/math.sqrt(avg_variance) # low variance = high score
        # ideal cluster has: Low variance w.r.t to cluster sizes and a decent (high) numbers of members in each cluster
        return avg_cluster_size + var_score
        
    def simple_diverse_hybrid(self, k):
        # first label maximally diverse examples
        #
        # note that the examples are 'labeled' within the method; thus
        # the confusing situation wherein we return only those selected
        # by SIMPLE because those selected by label_maxim... are already
        # moved to the labeled set. The reason this method doesn't return
        # a list of ids to label is because it was easier to implement the
        # diversify strategy this way.
        self.label_maximally_diverse_set(k/2, initial_train_set = False)
        # now run simple
        return self.SIMPLE(k/2)
        

    def kff(self, k):
        self.label_maximally_diverse_set(k, use_kff = True)
        
    def label_maximally_diverse_set(self, k, initial_train_set=False, stacked=False, only_positives=False, use_kff = False):
        '''
        Returns the instance numbers for the k most diverse examples (selected greedily)
        
        '''
        # first, label one example at random 
        if initial_train_set:
            self.label_at_random(1)
            k = k-1
            self.rebuild_models()
        
        data = self.unlabeled_datasets[0]
        
        already_labeled_set = self.labeled_datasets[0]
        if only_positives:
            already_labeled_set = self.labeled_datasets[0].get_minority_examples()
            
        model = self.models[0]
        if stacked:
            data= self._meta_dataset(self.unlabeled_datasets)
            model = self._get_stacked_meta_model(undersample_first=True)
            
        for step in range(k):
            print step
            
            #if not step%10:
            #    print "on step %s" % step
            # add examples iteratively, selecting the most diverse w.r.t. to the examples already selected in each step.
            x = data.instances[0]
            most_diverse_id = x.id
            most_diverse_score = self._compute_div(model, already_labeled_set, x) if not use_kff else self._get_dist_from_l(model, already_labeled_set, x)
            
            for x in data.instances[1:]:
                # now iterate over the remaining unlabeled examples
                cur_score = None
                if not use_kff:
                    cur_score = self._compute_div(model, already_labeled_set, x)
                else:
                    cur_score = self._get_dist_from_l(model, already_labeled_set, x)
                
                if cur_score > most_diverse_score:
                    most_diverse_score = cur_score
                    most_diverse_id = x.id

            # now label the most diverse example, if this is the initial train set 
            self.label_instances_in_all_datasets([most_diverse_id])
        print "building models..."
        self.rebuild_models()
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
    
    
        
    def euclid_dist(self, x, y):
        if not (x.id,y.id) in self.euclid_hash:
            sum = 0.0
            xpoint, ypoint = x.point, y.point
            for coord in set(xpoint.keys() + ypoint.keys()):
                if not coord in xpoint.keys():
                    sum += ypoint[coord]**2
                elif not coord in ypoint.keys():
                    sum += xpoint[coord]**2
                else:
                    sum += abs(xpoint[coord] - ypoint[coord])**2
            self.euclid_hash[(x.id, y.id)] = math.sqrt(sum)
        return self.euclid_hash[(x.id, y.id)]
            
              
    def _compute_cos(self, model, x, y):
        if not (x.id, y.id) in self.div_hash:
            self.div_hash[(x.id, y.id)] = model.compute_cos_between_examples(x.point, y.point)
        return self.div_hash[(x.id, y.id)]
    
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
        
        
    def stacked_SIMPLE(self, k):
        '''
        SIMPLE via stacks over the models. This creates 'meta' feature vec
        '''
        print "generating stacked examples..."
        meta_unlabeled_examples = self._meta_dataset(self.unlabeled_datasets)
        # now build a model over the meta_labeled_examples (ie, the stacked predictions)
        print "done. building a model over the (labeled) stacked instances..."
        meta_model = self._get_stacked_meta_model(undersample_first=True)
        print "(meta) model built. running simple..."
        # now pick instances to label
        return self._SIMPLE(meta_model, meta_unlabeled_examples, k)
      
        
    def _get_stacked_meta_model(self, undersample_first=False):
        '''
        Builds a 'stacked' model using the raw prediction values from the labeled data belonging
        to this instance
        '''
        meta_labeled_examples = self._meta_dataset(self.labeled_datasets)
        copied_dataset = meta_labeled_examples.copy()
        if undersample_first:
            k = meta_labeled_examples.number_of_majority_examples() - meta_labeled_examples.number_of_minority_examples()
            print "(stacked) removing %s majority instances" % k
            copied_dataset.undersample(k)

        samples, labels = copied_dataset.get_samples_and_labels()
        param = svm_parameter(weight=[1, 1000])
        problem = svm_problem(labels, samples)
        param.C, param.gamma = grid_search(problem, param, sens_only=True)
        meta_model = svm_model(problem, param)
        return meta_model
        
        
    def _meta_dataset(self, datasets):
        meta_instances = []
        for x_index in range(len(datasets[0].instances)):
            meta_point = []
            for model_index in range(len(self.models)):
                # iterate over the model indices -- equivalent to the feature space indices
                # and add each model's prediction for x to the meta feature vector
                m = self.models[model_index]
                x = datasets[model_index].instances[x_index]
                #meta_point.append(m.distance_to_hyperplane(x.point))
                #print m.predict_values_raw(x.point)
                try:
                    meta_point.append(m.predict_values_raw(x.point)[0])
                except:
                    pdb.set_trace()
            # the instance class wants a dictionary (i.e., sparse) formatted point
            # mapping coodinates to values
            coordinates = range(len(self.models))
            dict_point = dict(zip(coordinates, meta_point))
            # just use the first feature space point to get the label and id of x, which
            # will be the same in all feature spaces
            x = datasets[0].instances[x_index]
            meta_instances.append(dataset.instance(x.id, dict_point, x.label))
            
        return dataset.dataset(meta_instances)

        
    def SIMPLE(self, k):
        '''
        Returns the instance numbers for the k unlabeled instances closest the hyperplane.
        '''
        # just use the first dataset for now....
        # TODO implement coin flip, etc
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
            
            #print "using RBF KERNEL"
            #param.kernel_type = RBF
            #param.degree = 2   
            # find C, gamma parameters for each model
            #print "finding optimal C, gamma parameters..."
            #param.C, param.gamma = grid_search(problem, param)
            #print "C:%s; gamma:%s" % (param.C, param.gamma)

            self.models.append(svm_model(problem, param))
        print "done."      


        