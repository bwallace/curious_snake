import pdb
import simple_svm_learner
from simple_svm_learner import *


class PALLearner(SimpleLearner):
        def __init__(self, unlabeled_datasets = [], models=None, undersample_before_eval = False, 
                                epsilon = 0.075, kappa = 8, label_at_least = 0, win_size=3):
            #
            # call the SimpleLearner constructor to initialize various globals
            SimpleLearner.__init__(self, unlabeled_datasets=unlabeled_datasets, models=models,
                                                    undersample_before_eval = undersample_before_eval)
            
            print "PAL learner: switching query function to Random!"
            self.query_function = self.get_random_unlabeled_ids 
            self.name = "PAL"
            self.kappa = kappa
            self.epsilon = epsilon
            self.label_at_least = label_at_least
            self.switched_to_exploit = False
            self.win_size = win_size
            
            # here we instantiate two lists with length equal to the number
            # of feature spaces. 
            #
            # datasets_converged holds booleans indicating
            # whether or not individual feature spaces have converged (i.e.,
            # their diversities have stabilized w.r.t. epsilon and kappa) --
            # so datasets_converged[i] tells us whether we need to keep checking
            # feature space i.
            self.datasets_converged = [False for d in self.unlabeled_datasets]
            # diversity_scores maintains lists for each feature space
            # of the observed diversity scores, as computed after new positives
            # are discovered.
            self.diversity_scores = [[] for d in self.unlabeled_datasets]
            self.observed_minorities = [[] for d in self.unlabeled_datasets]
        
        def active_learn(self, num_examples_to_label, batch_size=5):
            #
            # call the base active learn method
            #
            BaseLearner.active_learn(self, num_examples_to_label, batch_size=batch_size)
            labeled_so_far = len(self.labeled_datasets[0])
            N = labeled_so_far + len(self.unlabeled_datasets[0])
            if not self.switched_to_exploit and labeled_so_far >= self.label_at_least * N:
                self.check_if_we_should_switch_to_exploit()
            
        def check_if_we_should_switch_to_exploit(self):
            already_observed_ids = [minority.id for minority in self.observed_minorities[0]]
            for i, dataset in enumerate(self.labeled_datasets):
                if not self.datasets_converged[i]:
                    # first update the diversity scores
                    
                    new_minorities = [minority for minority in self.labeled_datasets[i].get_minority_examples() 
                                                if minority.id not in already_observed_ids]
                    for new_minority in new_minorities:
                        # first add the newly observed minority to the list (of observed minorities)
                        self.observed_minorities[i].append(new_minority)
                        # now compute the diversity score
                        div_score = self.compute_div_score(model_index=i)
                        # now append the diversity score to the list (of diversity scores)
                        self.diversity_scores[i].append(div_score)        
                        
                    # now that we've calculated and appended the scores for
                    # each newly labeled minority instance, check if we can switch
                    
                    # note that the denominator (x) is constant (step_size), 
                    # so we can ignore it here
                    dydxs = _numerical_deriv(self.diversity_scores[i])
                    smoothed = [abs(div) for div in _median_smooth(dydxs, window_size=self.win_size)]
                    
                    max_change = 1.0
                    if len(smoothed) >= self.kappa + self.win_size:
                        max_change = max(smoothed[-self.kappa:])
            
                    if max_change <= self.epsilon:
                        print "\nlearner %s: feature space %s has converged after %s labels" % (self.name, self.labeled_datasets[i].name,
                                                                                                                            len(self.labeled_datasets[i]))
                        self.datasets_converged[i] = True
                
                # finally, check how many feature spaces have converged
                # we switch to exploitation if more than half have (this is arbitrary)
                if self.datasets_converged.count(True) >= len(self.labeled_datasets) :
                    print "learner %s: more than half of the feature spaces have stabilized: switching to exploit!" % self.name
                    self.query_function = self.SIMPLE
                    self.switched_to_exploit = True

        def compute_div_score(self, model_index=0):
            '''
            computes a diversity score over the positive labeled examples
            '''
            div_sum, pwise_count = 0.0, 0.0
            min_instances = self.observed_minorities[model_index]
            if len(min_instances) == 1:
                # only one instance
                return 0
            
            for instance_1 in min_instances:
                for instance_2 in [inst for inst in min_instances if inst.id != instance_1.id]:
                    div_sum+=self._compute_cos(model_index, instance_1, instance_2)
                    pwise_count+=1.0

            # average pairwise cosine 
            return div_sum / pwise_count
        
def _median_smooth(X, window_size=3):
    smoothed = []
    window_index = 0
    for i in range(len(X)):
        if i+1 <= window_size/2.0:
            smoothed.append(X[i])
        else:
            smoothed.append(_median(X[window_index:window_index+window_size]))
            window_index += 1
    return smoothed
    
            
def _median(X):
    median = None
    if len(X) % 2:
      # odd number of elements
      median = float(X[len(X)/2])
    else:
       # Number of elements in data set is even.
       mid_point = len(X)/2   
       median = (X[mid_point-1] + X[mid_point])/2.0
    return median
  
def _numerical_deriv(X):
    dxs = [1]
    diffs = [X[i+1] - X[i] for i in range(len(X)-1)]
    dxs.extend(diffs)
    return dxs