'''

	Byron C Wallace
	Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
	
	curious_snake.py
	--
	This module is for running experiments.
	
    ... Now for some legal stuff.
    ----
    
    <legal>
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
        * Neither the name of the <organization> nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY <copyright holder> ''AS IS'' AND ANY
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
    </legal>
'''

import random
import pdb
import math
import dataset
import learner 

def run_experiments_hold_out(data_paths, outpath, hold_out_p = .25,  upto=1000, step_size = 25, 
                                initial_size = 2, num_runs=10, eps=.0003, kappa=8, win_size=3, at_least_p = .05, for_eval = None):
    for run in range(num_runs):
        already_exhausted = False
        print "\n********\non run %s" % run
        print data_paths
        cur_size = initial_size

        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        
        datasets = [dataset.build_dataset_from_file(f) for f in data_paths]
        
        d_for_eval = None
        if for_eval:
            d_for_eval = dataset.build_dataset_from_file(for_eval)
        else:
            d_for_eval = datasets[0].copy()

        total_num_examples = len(datasets[0].instances)
        hold_out_size = int(hold_out_p * total_num_examples)
        
        print "total minorities: in *whole* dataset: %s" % len(datasets[0].get_minority_examples().instances)
        test_instances = random.sample(datasets[0].instances, hold_out_size)
        test_set_instance_ids = [inst.id for inst in test_instances]
        test_datasets = []
        
        # in case we want to use level 2 labels
        test_lbls = d_for_eval.remove_instances(test_set_instance_ids)
        for d in datasets:
            cur_test_dataset = dataset.dataset(d.remove_instances(test_set_instance_ids))
            for inst, gold_standard_inst in zip(cur_test_dataset.instances, test_lbls):
                inst.label = inst.real_label = gold_standard_inst.label
                
            test_datasets.append(cur_test_dataset)
        
        print "removed %s out of %s instances for test set, containing %s minority instances." % (hold_out_size, total_num_examples, len(test_datasets[0].get_minority_examples().instances))
        
        test_set_out = open("%s//test_set_%s" % (outpath, run), 'w')
        test_set_out.write(test_datasets[0].get_points_str())
        test_set_out.close()
        
        # U -- train set
        U_out = open("%s//U_%s" % (outpath, run), 'w')
        U_out.write(datasets[0].get_points_str())
        U_out.close()
        
        fmin = open("%s//minorities_in_test_%s.txt" % (outpath, run), 'w')
        fmin.write(test_datasets[0].get_minority_examples().get_points_str())
        fmin.close()
        
        active_learner = learner.learner([d.copy() for d in datasets])
        random_learner = learner.learner([d.copy() for d in datasets])
        al_star_learner = learner.learner([d.copy() for d in datasets])
        osugi_learner = learner.learner([d.copy() for d in datasets])
        osugi_learner.X_for_osugi = X_for_osugi
        osugi_learner.is_osugi = True
        
        activeout = open("%s//active_%s.txt" % (outpath, run), 'w')
        div_c_active =  open("%s//div_curve_active_%s.txt" % (outpath, run), 'w')
        al_star_out = open("%s//al_star_%s.txt" % (outpath, run), 'w')
        div_c_star = open("%s//div_curve_star_%s.txt" % (outpath, run), 'w')
        randomout =  open("%s//random_%s.txt" % (outpath, run), 'w')
        div_c_random = open("%s//div_curve_random_%s.txt" % (outpath, run), 'w')
        osugi_out = open("%s//osugi_%s.txt" % (outpath, run), 'w')
        div_c_osugi = open("%s//div_curve_osugi_%s.txt" % (outpath, run), 'w')
        
        init_ids = random_learner.pick_balanced_initial_training_set(cur_size/2)
        al_star_learner.label_instances_in_all_datasets(init_ids)
        active_learner.label_instances_in_all_datasets(init_ids)
        osugi_learner.label_instances_in_all_datasets(init_ids)
        
        #
        # Build initial models
        #
        active_learner.rebuild_models(undersample_first=True)
        random_learner.rebuild_models(undersample_first=True)
        al_star_learner.rebuild_models(undersample_first=True)
        osugi_learner.rebuild_models(undersample_first=True)
        

        print "\nACTIVE:"
        print "active learner has %s labeled examples" % len(active_learner.labeled_datasets[0].instances)
        active_results = learner.evaluate_learner_with_holdout(active_learner, test_datasets)
        write_out_results(active_results, activeout, cur_size)

        

        print "\nRANDOM:"
        print "random learner has %s labeled examples" % len(random_learner.labeled_datasets[0].instances)
        random_results = learner.evaluate_learner_with_holdout(random_learner, test_datasets)
        write_out_results(random_results, randomout, cur_size)

        
        print "\nSIMPLE*:"
        print "simple* learner has %s labeled examples" % len(al_star_learner.labeled_datasets[0].instances)
        al_star_results = learner.evaluate_learner_with_holdout(al_star_learner, test_datasets)
        write_out_results(al_star_results, al_star_out, cur_size)
        
        
        print "\nOSUGI:"
        print "osugi learner has %s labeled examples" % len(osugi_learner.labeled_datasets[0].instances)
        osugi_results = learner.evaluate_learner_with_holdout(osugi_learner, test_datasets)
        write_out_results(osugi_results, osugi_out, cur_size)
        
        
        last_pos_c = 0
        former_pos_set = []
        first_iter = True

        switched_to_simple = False
        outf = open("%s//pos_diversities_%s.csv" % (outpath, run), "w")
        delta_pos = open("%s//pos_deltas_%s.csv" % (outpath, run), "w") 
        minority_diversity_scores = []
        al_star_q_function = al_star_learner.label_at_random # start with random labeling
        iter_num = 0
    
        while cur_size <=upto:
            al_star_learner.write_out_minorities("%s//al_star_mins_labeled_%s.csv" % (outpath, run))
            random_learner.write_out_minorities("%s//random_mins_labeled_%s.csv" % (outpath, run))
            active_learner.write_out_minorities("%s//active_mins_labeled_%s.csv" % (outpath, run))
            osugi_learner.write_out_minorities("%s//osugi_mins_labeled_%s.csv" % (outpath, run))
            
            al_star_learner.write_out_labeled_data("%s//al_star_labeled_%s.csv" % (outpath, run))
            random_learner.write_out_labeled_data("%s//random_labeled_%s.csv" % (outpath, run))
            active_learner.write_out_labeled_data("%s//active_labeled_%s.csv" % (outpath, run))
            osugi_learner.write_out_labeled_data("%s//osugi_labeled_%s.csv" % (outpath, run))
            print "\n\n***using %s examples out of %s***" % (cur_size, upto)

            pos_set = al_star_learner.labeled_datasets[0].get_minority_examples()
            cur_pos_for_al = len(pos_set.instances)
            
            cur_change = cur_pos_for_al - last_pos_c 
            last_pos_c = cur_pos_for_al
            print "min change: %s" % cur_change
            delta_pos.write("%s" % cur_change) 
            print cur_change
            if cur_change > 0 or len(minority_diversity_scores) == 0:
                new_pos = [pos for pos in pos_set.instances if not pos in former_pos_set]
                print "\n length of new pos: %s" % len(new_pos)
                for pos in new_pos:
                    div_score = 0
                    former_pos_set.append(pos)
                    div_score = compute_div_score(former_pos_set, al_star_learner)
                    minority_diversity_scores.append(div_score)
                    print "div score over %s instances, with %s positives: %s" % (len(al_star_learner.labeled_datasets[0].instances), len(pos_set.instances), div_score)
                    outf.write("%s, %s, %s\n" % (len(al_star_learner.labeled_datasets[0].instances), len(pos_set.instances), div_score))
            else:
                print "\n\n\n****No new positives found-- not updating diversity score!***\n\n"
            former_pos_set = pos_set.instances
            print "diversity scores:"
            print minority_diversity_scores
            print "\n"

            #
            # Write out the diversity curves
            #
            div_c_random.write(str(compute_div_score(random_learner.labeled_datasets[0].get_minority_examples().instances, random_learner)) + ",")
            div_c_active.write(str(compute_div_score(active_learner.labeled_datasets[0].get_minority_examples().instances, active_learner)) + ",")
            div_c_star.write(str(compute_div_score(al_star_learner.labeled_datasets[0].get_minority_examples().instances, al_star_learner)) + ",")
            div_c_osugi.write(str(compute_div_score(osugi_learner.labeled_datasets[0].get_minority_examples().instances, osugi_learner)) + ",")
            
            # active learn iteration
            if first_iter:
                tmp_step_size = step_size-cur_size
                random_learner.active_learn(tmp_step_size, query_function = random_learner.label_at_random, rebuild_models_at_each_iter = False, num_to_label_at_each_iteration=1)
                active_learner.active_learn(tmp_step_size, num_to_label_at_each_iteration=1)
                osugi_learner.active_learn(tmp_step_size, num_to_label_at_each_iteration=1)
                al_star_learner.active_learn(tmp_step_size, query_function = al_star_q_function, rebuild_models_at_each_iter = False, num_to_label_at_each_iteration=1)
                cur_size = step_size
                first_iter=False
            else:
                random_learner.active_learn(step_size, num_to_label_at_each_iteration = 5, query_function = random_learner.label_at_random, rebuild_models_at_each_iter = False)
                active_learner.active_learn(step_size, num_to_label_at_each_iteration=5)
                osugi_learner.active_learn(step_size, num_to_label_at_each_iteration=1)
                al_star_learner.active_learn(step_size, query_function = al_star_q_function, num_to_label_at_each_iteration=5)
                cur_size+=step_size


            outf.write("%s,\n" % minority_diversity_scores[-1])
            delta_ys = numerical_deriv(minority_diversity_scores)
            delta_xs = [step_size for y in range(len(delta_ys))]
            dydxs = [dy/dx for dy, dx in zip(delta_ys, delta_xs)]
            print "\n\ndysdxs"
            print dydxs
            print "\n"
            # now smooth them
            smoothed = [abs(div) for div in median_smooth(dydxs, window_size=win_size)]
            max_change = 1.0
            if len(smoothed) >= kappa + win_size:
                max_change = max(smoothed[-kappa:])

            print "\nsmoothed[:-KAPPA]\n"
            print smoothed[-kappa:]
            print "\n"
            print "\nmax change: %s" % max_change
            num_labeled_so_far = len(al_star_learner.labeled_datasets[0].instances)
            if abs(max_change) <= eps and num_labeled_so_far >= at_least_p*total_num_examples and not switched_to_simple:
                print "\n\nSWITCHED SIMPLE* TO ACTIVE... labeled so far: %s; number of minorities discovered: %s" % (num_labeled_so_far, len(active_learner.labeled_datasets[0].get_minority_examples().instances))
                al_star_q_function = al_star_learner.SIMPLE
                switched_to_simple = True
                f = open("%s//switched_at_%s.txt"  % (outpath, run), "w")
                f.write(str(num_labeled_so_far))
                f.close()


    
            if margin_exhausted and not already_exhausted:
                already_exhausted=True
                fmargin = open("%s//margin_exhausted_%s.txt" % (outpath, run), "w")
                print "\n\nMARGIN EXHAUSTED @ %s examples\n\n" % num_labeled_so_far
                fmargin.write(str(num_labeled_so_far))
                fmargin.close()
            elif margin_exhausted:
                print "already exhuasted"
            else:
                print "...not exhuasted."

            
            active_learner.rebuild_models(undersample_first=True)
            random_learner.rebuild_models(undersample_first=True)
            al_star_learner.rebuild_models(undersample_first=True)
            osugi_learner.rebuild_models(undersample_first=True)
            
            iter_num+=1
            
            # write the results out

            print "\nACTIVE:"
            print "active learner has %s labeled examples" % len(active_learner.labeled_datasets[0].instances)
            active_results = learner.evaluate_learner_with_holdout(active_learner, test_datasets)
            write_out_results(active_results, activeout, cur_size)

            
            print "\nRANDOM:"
            print "random learner has %s labeled examples" % len(random_learner.labeled_datasets[0].instances)
            random_results = learner.evaluate_learner_with_holdout(random_learner, test_datasets)
            write_out_results(random_results, randomout, cur_size)

       
            print "\nOSUGI:"
            print "osugi learner has %s labeled examples" % len(osugi_learner.labeled_datasets[0].instances)
            print "p is %s" % (osugi_learner.osugi_p)
            osugi_results = learner.evaluate_learner_with_holdout(osugi_learner, test_datasets)
            write_out_results(osugi_results, osugi_out, cur_size)
        
            
            print "\nSIMPLE*:"
            if switched_to_simple:
                print "now using simple"
            else:
                print "still random sampling"
            print "simple* learner has %s labeled examples" % len(al_star_learner.labeled_datasets[0].instances)
            al_star_results = learner.evaluate_learner_with_holdout(al_star_learner, test_datasets)
            write_out_results(al_star_results, al_star_out, cur_size)


        # close files
        div_c_star.close()
        div_c_active.close()
        div_c_random.close()
        div_c_osugi.close()
        outf.close()
        activeout.close()
        randomout.close()
        al_star_out.close()
        osugi_out.close()
      

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
    
     
def write_out_results(results, outf, size):
    write_these_out = [ size, results["accuracy"], results["sensitivity"], results["npos"]]
    outf.write(",".join([str(s) for s in write_these_out]))
    outf.write("\n")