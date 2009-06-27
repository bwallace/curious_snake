'''
	Byron C Wallace
	Tufts Medical Center: Computational and Analytic Evidence Sythensis (tuftscaes.org)
	Curious Snake: Active Learning in Python with SVMs
	results_reporter.py
	---
	
	This module is for aggregating and reporting the output of experimental runs. It uses pylab
	to generate the standard 'learning curves' for each of the learners.
'''

import os
import pdb
try:
    import matplotlib.pyplot as plt
    import pylab
except:
    print '''whoops, results_reporter module can't load pylab library. you can still run your experiments -- 
                data will be written out to files -- but I can't plot your results. '''
    
def post_runs_report(base_path, learner_labels, n, metrics = ["num_labels", "accuracy", "sensitivity", "specificity"]):    
    '''
    Call this method when the analysis finishes. It:
        1) Averages the n runs
        2) Plots the results for each metric in metric and saves these to the 
        base_path directory. Note: Requires pylab! 
    '''        
    print "averaging results..."
    averages = avg_results(base_path, learner_labels, n, metrics) 
    print "done."
    
    # note that we skip the number of labels
    for metric in metrics[1:]:
        plot_metric_for_learners(averages, metric, output_path = os.path.join(base_path, metric) + ".pdf")
    
    print "plots generated"
    
    
def plot_metric_for_learners(results, metric, legend_loc = "lower right", x_key = "num_labels", 
                            output_path = None, show_plot = True):
    '''
    Uses pylab to generate a learning plot (number of labels v. metric of interest) for each of the learners.
    '''
    learner_names = results.keys()
    learner_names.sort()
    
    # we build a list of the lines plotted (the plot routine returns a line)
    # we need these for the legend.
    lines = []
    # clear the canvass
    pylab.clf()
    
    for learner in learner_names:
        lines.append(pylab.plot(results[learner]["num_labels"], results[learner][metric], 'o-'))
    
    pylab.legend(lines, learner_names, legend_loc, shadow = True)
    pylab.xlabel("Number of Labels")
    pylab.ylabel(metric)
    
    # if an output path was passed in, save the plot to it
    if output_path is not None:
        pylab.savefig(output_path, format="pdf")
    
    if show_plot:
        pylab.show()
        
    
def avg_results(base_path, learner_names, n, metrics, size_index=0):
    '''
    This method aggregates the results from the files output during the active learing simulation, building
    averaged time curves for each of the metrics and returning these in a dictionary.
    
    TODO make the metrics list a global member of curious_snake; use this to write out results; 
    
    n -- number of runs, i.e., number of files
    
    '''
    averaged_results_for_learners = {}
    for learner in learner_names:
        running_totals, sizes, num_steps = None, None, None
    
        for run in range(n):
            cur_run_results = _parse_results_file(os.path.join(base_path, learner + "_" + str(run) + ".txt"))
            if running_totals is None:
                # on the first pass through, we build an initial zero matrix to store our averages. we do this
                # here because we know how many steps there were (the length, or number of rows, of the first 
                #`cur_run_results' file)
                num_steps = len(cur_run_results)
                running_totals = []
                for step_i in range(num_steps):
                    running_totals.append([0.0 for metric in range(len(metrics))])
                sizes = [0.0 for step_i in range(num_steps)]

            for step_index in range(num_steps):
                for metric_index in range(len(metrics)):
                    running_totals[step_index][metric_index] += float(cur_run_results[step_index][metric_index])
                if run == 0:
                    # set the sizes on the first pass through (these will be the same for each run)
                    sizes[step_index] = float(cur_run_results[step_index][size_index])

        averages = []
        for metric_i in range(len(metrics)):
            cur_metric_avg = []
            for step_i in range(num_steps):
                cur_metric_avg.append(running_totals[step_i][metric_i] / float(n))
            averages.append(cur_metric_avg)

        averaged_results_for_learners[learner] = dict(zip(metrics, averages))

    return averaged_results_for_learners
             
def average(x):
    return float(sum(x)) / float(len(x))

def _parse_results_file(fpath):
    return_mat = []
    for l in open(fpath, 'r').readlines():
        return_mat.append([eval(s) for s in l.replace("\n", "").split(",")])
    return return_mat