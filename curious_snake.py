'''
############################################

	abstrackr.py
	Byron C Wallace
	Tufts Medical Center
	module for running experiments.

############################################
'''

import os
import sys
import random
import pdb
path_to_libsvm = os.path.join(os.getcwd(), "libsvm", "python")
sys.path.append(path_to_libsvm)
import svm
from svm import *
import numpy
import math
import dataset
import learner 
import numpy
import Stats
#from learner import learner
#data_paths = ["data//GREL"]
#data_paths = ["data//fbis_20.txt"]
#data_paths = ["data//G157"]
#data_paths = ["data//E143"] # tiny
#data_paths= ["data//fbis_111.txt"]
#data_paths= ["data//E61"] #no.
#data_paths= ["data//E71"]
#data_paths = ["data//C313"]
#data_paths = ["data//C312"]
#data_paths = ["data//fbis_20.txt"]
#data_paths = ["data//fbis_108_11.txt"]
#data_paths = ["data//Molecular-Sequence-Data.txt"]
#data_paths = ["data//la_national.txt"]
#data_paths = ["data//cocoa_gold.txt"]
#data_paths = ["data//news_12_1.txt"]
#data_paths = ["data/news_12.txt"]
#data_paths = ["data//news_221_108_273.txt"]
#data_paths = ["data//l1_keywords.txt", "data//l1_titles.txt", "data//l1_title_concepts.txt"]
#data_paths = ["data//fbis_202.txt"]
#data_paths = ["data//sati"]
def simulate_screening(training_size = 200):
    datasets = [dataset.build_dataset_from_file(f) for f in data_paths]
    larry = learner.learner(datasets)
    larry.pick_initial_training_set(training_size)
    larry.rebuild_models(undersample_first=True)
    return learner.evaluate_learner(larry)  

def empirical_best(data_paths, outpath, num_runs=10, hold_out_p=.25):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
        
    for run in range(num_runs):
        print "\n********\non run %s" % run
        print data_paths
        
        datasets = [dataset.build_dataset_from_file(f) for f in data_paths]
        total_num_examples = len(datasets[0].instances)
        hold_out_size = int(hold_out_p * total_num_examples)
        print "total minorities: in *whole* dataset: %s" % len(datasets[0].get_minority_examples().instances)
        test_instances = random.sample(datasets[0].instances, hold_out_size)
        test_set_instance_ids = [inst.id for inst in test_instances]
        test_datasets = []
        for d in datasets:
            test_datasets.append(dataset.dataset(d.remove_instances(test_set_instance_ids)))
        print "removed %s out of %s instances for test set, containing %s minority instances." % (hold_out_size, total_num_examples, len(test_datasets[0].get_minority_examples().instances))
        # just use all the data
        cur_size = len(datasets[0].get_minority_examples().instances)*2
        larry = learner.learner([d.copy() for d in datasets])
        larry.label_all_data()
        larry.rebuild_models(undersample_first=True, undersample_cleverly=False)
        outf = open("%s//all_positives_%s.txt" % (outpath, run), 'w')
        results = learner.evaluate_learner_with_holdout(larry, test_datasets)
        write_out_results(results, outf, cur_size)
        outf.close()
        
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
        
            
        X_for_osugi = datasets[0].get_samples()
        
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

            #
            # # # FOR EXPERIMENTATION ONLY # # # MOVE ME I DONT BELONG HERE
            #

            closest_sv = svmc.distance_of_closest_SV(active_learner.models[0].model)

            margin_exhausted = True
            for x in active_learner.unlabeled_datasets[0].instances:
                cur_d = abs(active_learner.models[0].distance_to_hyperplane(x.point)) 
                if cur_d  <= closest_sv and not already_exhausted:
                    margin_exhausted = False
                    print "not exhausted!"
                    print "cur distance: %s, closest sv: %s" % (cur_d, closest_sv)
                    break
                else:
                    pass
    
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
      

 
def run_experiments(upto=1000, step_size = 25, initial_size = 2, num_runs=10, eps=.0001, kappa=12, win_size = 1, at_least_p = .05):
    print "running experiments %s times using up to %s examples..." % (num_runs, upto)
    
    for run in range(num_runs):
        already_exhausted = False
        run = run
        activeout = open("output//active_%s.txt" % run, 'w')
        div_c_active =  open("output//div_curve_active_%s.txt" % run, 'w')
        #stackedout = open("output//stacked_%s.txt" % run, 'w')
        al_star_out = open("output//al_star_%s.txt" % run, 'w')
        #div_c_star = open("output//div_curve_star_%s.txt" % run, 'w')
        randomout =  open("output//random_%s.txt" % run, 'w')
        div_c_random = open("output//div_curve_random_%s.txt" % run, 'w')
       # osugi_out = open("output//osugi_%s.txt % run, 'w'")
        '''
        diverseout = open("output//diverse_%s.txt" % run, 'w')
        hybridout = open("output//hybrid_%s.txt" % run, 'w')
        stackedout = open("output//stacked_%s.txt" % run, 'w')
        '''
        
        
        print "\n********\non run %s" % run
        print data_paths
        cur_size = initial_size
        
        datasets = [dataset.build_dataset_from_file(f) for f in data_paths]
      
        print len(datasets[0].get_minority_examples().instances)
        total_num_examples = len(datasets[0].instances)
        
        active_learner = learner.learner(datasets)
        random_learner = learner.learner([d.copy() for d in datasets])
        al_star_learner = learner.learner([d.copy() for d in datasets])
      #  osugi_learner = learner.learner([d.copy() for d in datasets])
        
        #hybrid_learner = learner.learner([d.copy() for d in datasets])
        #active_learner = learner.learner(datasets)
        #stacked_learner = learner.learner([d.copy() for d in datasets])
        #random_learner = learner.learner([d.copy() for d in datasets])

        
        # 
        # Pick initial labeled datasets (bootstrap)
        #
        
        #random_learner.pick_initial_training_set(cur_size)
        #init_ids = [x.id for x in random_learner.labeled_datasets[0].instances]
        #pdb.set_trace()
        init_ids = random_learner.pick_balanced_initial_training_set(cur_size/2)
        
        #stacked_learner.pick_initial_training_set(cur_size/2)
        #stacked_learner.rebuild_models(undersample_first=True)
        #random_learner.pick_initial_training_set(cur_size)
        #random_learner.pick_balanced_initial_training_set(cur_size/2)
        random_learner.label_instances_in_all_datasets(init_ids)
        al_star_learner.label_instances_in_all_datasets(init_ids)
        active_learner.label_instances_in_all_datasets(init_ids)
      #  osugi_learner.label_instances_in_all_datasets(init_ids)
        
        '''
        diverse_learner.label_maximally_diverse_set(cur_size, initial_train_set=True)
        hybrid_learner.pick_initial_training_set(cur_size)
        stacked_learner.pick_initial_training_set(cur_size)
        '''
        
        #
        # Build initial models
        #
        active_learner.rebuild_models(undersample_first=True)
        random_learner.rebuild_models(undersample_first=True)
        al_star_learner.rebuild_models(undersample_first=True)
#       osugi_learner.rebuild_models(undersample_first=True)
        
        # write the results out
        print "results using bootstrap training sets"
        
        active_results = learner.evaluate_learner(active_learner)
        write_out_results(active_results, activeout, cur_size)
    
        random_results = learner.evaluate_learner(random_learner)
        write_out_results(random_results, randomout, cur_size)
        
        al_star_results = learner.evaluate_learner(al_star_learner)
        write_out_results(al_star_results, al_star_out, cur_size)
        
     #   osugi_results = learner.evaluate_learner(osugi_learner)
    #    write_out_results(osugi_results, osugi_out, cur_size)
        
        '''
        stacked_results = learner.evaluate_learner(stacked_learner)
        write_out_results(stacked_results,stackedout, cur_size)

        #diverse_results = learner.evaluate_learner(diverse_learner)
        #write_out_results(diverse_results, diverseout, cur_size)

        stacked_results = learner.evaluate_learner(stacked_learner)
        '''
        last_pos_c = 0
        #former_pos_set = al_star_learner.labeled_datasets[0].get_minority_examples().instances
        former_pos_set = []
        first_iter = True
       # cur_size+=step_size
        # now loop up to the upto value labeling examples at each step
        #hybrid_meth =  hybrid_learner.hyper_dist_SIMPLE_hybrid
        switched_to_simple = False
        outf = open("output//pos_diversities_%s.csv" % run, "w")
        delta_pos = open("output//pos_deltas_%s.csv" % run, "w") 
        minority_diversity_scores = []
        al_star_q_function = al_star_learner.label_at_random # start with random labeling
        iter_num = 0
        while cur_size <=upto:
            print "\n\n***using %s examples out of %s***" % (cur_size, upto)
            
            pos_set = active_learner.labeled_datasets[0].get_minority_examples()
            cur_pos_for_al = len(pos_set.instances)
            
            cur_change = cur_pos_for_al - last_pos_c 
            last_pos_c = cur_pos_for_al
            print "min change: %s" % cur_change
            #pdb.set_trace()
            delta_pos.write("%s" % cur_change) 
            print cur_change
            if cur_change > 0 or len(minority_diversity_scores) == 0:
                
                new_pos = [pos for pos in pos_set.instances if not pos in former_pos_set]
                for pos in new_pos:
                    div_score = 0
                    former_pos_set.append(pos)
                    div_score = compute_div_score(former_pos_set, al_star_learner)
                    minority_diversity_scores.append(div_score)
                    print "div score over %s instances, with %s positives: %s" % (len(al_star_learner.labeled_datasets[0].instances), len(pos_set.instances), div_score)
                    outf.write("%s, %s, %s\n" % (len(al_star_learner.labeled_datasets[0].instances), len(pos_set.instances), div_score))
            else:
                #pdb.set_trace()
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
            
            # active learn iteration
            if first_iter:
               # pdb.set_trace()
                tmp_step_size = step_size-cur_size
                random_learner.active_learn(tmp_step_size, query_function = random_learner.label_at_random, rebuild_models_at_each_iter = False, num_to_label_at_each_iteration=1)
                active_learner.active_learn(tmp_step_size, num_to_label_at_each_iteration=1)
                al_star_learner.active_learn(tmp_step_size, query_function = al_star_q_function, rebuild_models_at_each_iter = False, num_to_label_at_each_iteration=1)
                #hybrid_learner.active_learn(tmp_step_size, query_function = hybrid_learner.label_at_random, num_to_label_at_each_iteration=2) # active learn first!
                cur_size+=tmp_step_size
                first_iter=False
            else:
                random_learner.active_learn(step_size, num_to_label_at_each_iteration = 5, query_function = random_learner.label_at_random, rebuild_models_at_each_iter = False)
                active_learner.active_learn(step_size, num_to_label_at_each_iteration=5)
                al_star_learner.active_learn(step_size, query_function = al_star_q_function, num_to_label_at_each_iteration=5)
                cur_size+=step_size


            #
            #
            #
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
                #pdb.set_trace()
            
            print "\nsmoothed[:-KAPPA]\n"
            print smoothed[-kappa:]
            print "\n"
            print "\nmax change: %s" % max_change
            num_labeled_so_far = len(al_star_learner.labeled_datasets[0].instances)
            if abs(max_change) <= eps and num_labeled_so_far >= at_least_p*total_num_examples and not switched_to_simple:
                print "\n\nSWITCHED SIMPLE* TO ACTIVE... labeled so far: %s; number of minorities discovered: %s" % (num_labeled_so_far, len(active_learner.labeled_datasets[0].get_minority_examples().instances))
                al_star_q_function = al_star_learner.SIMPLE
                switched_to_simple = True
                f = open("switched_at.txt", "w")
                f.write(str(num_labeled_so_far))
                f.close()
                
            # # # FOR EXPERIMENTATION ONLY # # # MOVE ME I DONT BELONG HERE

            closest_sv = svmc.distance_of_closest_SV(active_learner.models[0].model)
            
            margin_exhausted = True
            for x in active_learner.unlabeled_datasets[0].instances:
                cur_d = abs(active_learner.models[0].distance_to_hyperplane(x.point)) 
                if cur_d  <= closest_sv and not already_exhausted:
                    margin_exhausted = False
                    print "not exhausted!"
                    print "cur distance: %s, closest sv: %s" % (cur_d, closest_sv)
                    break
                else:
                    pass
                    #print "cur d: %s; closest: %s" % (cur_d, closest_sv)
            
            if margin_exhausted and not already_exhausted:
                already_exhausted=True
                fmargin = open("output//margin_exhausted_%s.txt" % run, "w")
                print "\n\nMARGIN EXHAUSTED @ %s examples" % num_labeled_so_far
                fmargin.write(str(num_labeled_so_far))
                fmargin.close()
                #pdb.set_trace()
            elif margin_exhausted:
                print "already exhuasted"
            else:
                print "...not exhuasted."
           # outf.write("%s, %s\n" % (cur_change, cur_pos_for_al))
            #pdb.set_trace()
            active_learner.rebuild_models(undersample_first=True)
            random_learner.rebuild_models(undersample_first=True)
            al_star_learner.rebuild_models(undersample_first=True)
            #hybrid_learner.rebuild_models(undersample_first=True, include_synthetics=True)
        
            #stacked_learner.rebuild_models(undersample_first=True)
            #diverse_learner.rebuild_models(undersample_first=True)
           # hybrid_learner.rebuild_models(undersample_first=True)
         
            
            # write the results out
            print "\nACTIVE:"
            print "active learner has %s labeled examples" % len(active_learner.labeled_datasets[0].instances)
            active_results = learner.evaluate_learner(active_learner)
            write_out_results(active_results, activeout, cur_size)
            
            print "\nRANDOM:"
            print "random learner has %s labeled examples" % len(random_learner.labeled_datasets[0].instances)
            random_results = learner.evaluate_learner(random_learner)
            write_out_results(random_results, randomout, cur_size)
         
            print "\nSIMPLE*:"
            #pdb.set_trace()
            if switched_to_simple:
                print "now using simple"
            else:
                print "still random sampling"
            print "simple* learner has %s labeled examples" % len(al_star_learner.labeled_datasets[0].instances)
            al_star_results = learner.evaluate_learner(al_star_learner)
            write_out_results(al_star_results, al_star_out, cur_size)
            
            iter_num+=1
            
            
        div_c_star.close()
        div_c_active.close()
        div_c_random.close()
        outf.close()
        activeout.close()
        randomout.close()
        al_star_out.close()
        #hybridout.close()
        '''
        diverseout.close()
        stackedout.close()
        '''
def compute_div_score(X, learner):
    div_sum = 0.0
    pwise_count = 0
    #for p1 in X.instances:
    for p1 in X:
        #for p2 in [y for y in X.instances if y.id != p1.id]:
        for p2 in [y for y in X if y.id != p1.id]:
            div_sum+=learner._compute_cos(learner.models[0], p1, p2)
            pwise_count+=1.0
    if pwise_count == 0:
        # only one instance
        return 0
    return div_sum / pwise_count
  
def median_smooth(X, window_size=3):
    Y = []
    window_index = 0
    for i in range(len(X)):
        if i+1 <= window_size/2.0:
            Y.append(X[i])
        else:
            Y.append(Stats.Statistics(X[window_index:window_index+window_size]).median)
            window_index += 1
    return Y   

def closest_labeled_point(dataset, model, sign):
    points = None
    if sign > 0:
        points = dataset.get_majority_examples()
    else:
        points = dataset.get_minority_examples()
    min_distance = min([model.distance_to_hyperplane(p) for p in points])
    return min_distance
        
    
def numerical_deriv(X):
    dxs = [1]
    diffs = [X[i+1] - X[i] for i in range(len(X)-1)]
    dxs.extend(diffs)
    return dxs
     
def write_out_results(results, outf, size):
    write_these_out = [ size, results["accuracy"], results["sensitivity"], results["npos"]]
    outf.write(",".join([str(s) for s in write_these_out]))
    outf.write("\n")