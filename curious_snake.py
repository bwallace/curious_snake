'''
	Byron C Wallace
	Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
	Curious Snake: Active Learning in Python with SVMs
	curious_snake.py
	--
	This module is for running experiments to compare active learning strategies. It uses the active learning framework.
	See the in-line documentation for examples.
	
	Two general notes: 
	
	(1) Curious Snake was originally written for a scenario in which multiple feature spaces
    were being exploited, thus pluralizing many of the attributes in this class. For example, 
    *lists* of unlabeled_datasets and models are kept. If you only have one feature space that you're interested
     in, as is often the case, simply pass around unary lists. 
     
     (2) It is assumed throughout the active learning is being done over binary datasets.
     
    ... Now for some legal stuff.
    ----
    CuriousSnake is distributed under the modified BSD licence
    Copyright (c)  2009,  byron c wallace
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of Tufts Medical Center nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY byron c wallace 'AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.    
    
    The files comprising the libsvm library are also under the modified BSD and are:
    
    Copyright (c) 2000-2008 Chih-Chung Chang and Chih-Jen Lin
    All rights reserved.
'''

import random
import pdb
import os
import math
import dataset
import base_learner
import simple_learner
import svm

def run_experiments_hold_out(data_paths, outpath, hold_out_p = .25,  datasets_for_eval =None, upto=None, step_size = 25, 
                                                                  initial_size = 2, batch_size = 5,  pick_balanced_initial_set = True, num_runs=10):
    '''
    This method demonstrates how to use the active learning framework, and is also a functional routine for comparing learners. Basically,
    a number of runs will be performed, the active learning methods will be evaluated at each step, and results will be reported. The results
    for each run will be dumped to a text files, which then can be combined (e.g., averaged), elsewhere.
    
    @parameters
    data_paths -- this is either a list (pointing to multiple feature spaces for the same instances) or a string pointing to a single data file (this will be
                                the typical case). e.g., data_paths = "mydata.txt". curious_snake uses a sparse-formated weka-like format, documented elsewhere.
    outpath -- this is a directory under which all of the results will be dumped.
    hold_out_p -- the hold out percentage, i.e., how much of your data will be used for evaluation. you can ignore this is you're providing your own    
                                  dataset(s) for evaluation (i.e., datasets_for_eval is not None)'.
    datasets_for_eval -- use this is you have datasets you want to use for testing -- i.e., to specify your hold out set independent of the data
                                                in data_paths. 
    upto -- active learning will stop when upto examples have been labeled. if this is None, upto will default to the total unlabeled pool available
    initial_size -- the size of 'bootstrap' set to use prior to starting active learning (for the initial models)
    batch_size -- the number of examples to be labeled at each iteration in active learning -- optimally, 1
    step_size -- results will be reported every time another step_size examples have been labeled
    pick_balanced_initial_set -- if True, the initial train dataset will be built over an equal number (initial_size/2) of both classes.
    num_runs -- this many runs will be performed
    '''
    
    for run in range(num_runs):
        print "\n********\non run %s" % run
        print data_paths
        cur_size = initial_size # set to initial size for first iteration

        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        
        # if a string (pointing to a single dataset) is passed in, box it in a list
        data_paths = box_if_string(data_paths)
        
        datasets = [dataset.build_dataset_from_file(f) for f in data_paths]
        total_num_examples = len(datasets[0].instances)
        
        test_datasets = []
        if datasets_for_eval is not None:
            # if a test set is specified, use it.
            datasets_for_eval = box_if_string(datasets_for_eval)
            test_datasets = [dataset.build_dataset_from_file(f) for f in datasets_for_eval]
            if upto is None:
                upto = total_num_examples
        else:
            # other wise, we copy the first (even if there multiple datasets, it won't matter, as we're just using 
            #the labels) and pick random examples
            d_for_eval = None
            d_for_eval = datasets[0].copy()
            hold_out_size = int(hold_out_p * total_num_examples)
            test_instances = random.sample(datasets[0].instances, hold_out_size)
            test_set_instance_ids = [inst.id for inst in test_instances]
            test_lbls = d_for_eval.remove_instances(test_set_instance_ids)
            for d in datasets:
                cur_test_dataset = dataset.dataset(d.remove_instances(test_set_instance_ids))
                for inst, gold_standard_inst in zip(cur_test_dataset.instances, test_lbls):
                    inst.label = inst.real_label = gold_standard_inst.label
                    
                test_datasets.append(cur_test_dataset)
            # if no upper bound was passed in, use the whole pool U
            if upto is None:
                upto = total_num_examples - hold_out_size

        print "using %s out of %s instances for test set" % (hold_out_size, total_num_examples)
        
        #
        # Here is where learners can be added for comparison
        #
        learners = [base_learner.BaseLearner([d.copy() for d in datasets]), simple_learner.SimpleLearner([d.copy() for d in datasets])]
        pdb.set_trace()
        output_files = [open("%s//%s_%s.txt" % (outpath, learner.name, run), 'w') for learner in learners]

        # we arbitrarily pick the initial ids from the first learner; this doesn't matter, as we just use the instance ids
        initial_f = learners[0].get_random_unlabeled_ids 
        init_size = cur_size
        if pick_balanced_initial_set:
            initial_f = learners[0].pick_balanced_initial_training_set
            init_size = int(cur_size/2.0) # equal number from both classes
            
        # Again, you could call *.initial_f on any learner -- it just returns the ids to label initially. these should
        # be the same for all learners.
        init_ids =initial_f(init_size)
        
        # label instances and build initial models
        for learner in learners:
            learner.label_instances_in_all_datasets(init_ids)
            learner.rebuild_models()
            
        #
        # report initial results, to console and file.
        #
        report_results(learners, test_datasets, cur_size, output_files)
                

        while cur_size <=upto:
            #
            # here's the main active learning loop
            #
            print "\n\n***using %s examples out of %s***" % (cur_size, upto)
            for learner in learners:
                learner.active_learn(step_size, num_to_label_at_each_iteration = batch_size)
                            
            cur_size+=step_size

            report_results(learners, test_datasets, cur_size, output_files)

        # close files
        for output_file in output_files:
            output_file.close()
        
     
def report_results(learners, test_datasets, cur_size, output_files):
    ''' 
    Writes results for the learners, as evaluated over the test_dataset(s), to the console and the parametric
    output files.
    '''
    learner_index = 0
    for learner in learners:
        print "results for %s @ %s labeled examples:" % (learner.name, len(learner.labeled_datasets[0].instances))
        results = evaluate_learner_with_holdout(learner, test_datasets)
        write_out_results(results, output_files[learner_index], cur_size)
        learner_index+=1
     
     
def box_if_string(s):
    '''
    If s is a string, returns a unary list [s]
    '''
    if type(s) == type(""):
        # then a string, presumably pointing to a single data file, was passed in
        return [s]
    return s
        
def evaluate_learner_with_holdout(learner, test_sets):
    '''
    If you're not considering a "finite pool" problem, this is really the correct way to evaluate the trained classifiers. 
    
    @params
    learner -- the learner to be evaluated
    test_sets -- the set(s) of examples to be used for evaluation. if there are multiple, it is assumed that they correspond to multiple feature
                            spaces, thus they will have to be cominbed somehow. The 'predict' method in the learner class(es) handles this, see that 
                            method in, e.g., base_learner, for more.
    '''
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
    
     
def write_out_results(results, outf, size):
    write_these_out = [ size, results["accuracy"], results["sensitivity"], results["npos"]]
    outf.write(",".join([str(s) for s in write_these_out]))
    outf.write("\n")