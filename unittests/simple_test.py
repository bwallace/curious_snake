import os
import sys
import unittest
# first resolve path/imports.
# todo: the path resolution stuff is kind of hacky right now
base_path = os.path.abspath('..')
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "learners"))
sys.path.append(os.path.join(base_path, "learners", "svm_learners", "libsvm", "python"))
import svm
from svm import *
import learners.svm_learners.simple_svm_learner as simple_svm_learner
import dataset

class TestSimpleLearner(unittest.TestCase):
    def setUp(self):
        instances = [dataset.Instance(0, {0:.2, 1:1.0}, label=-1.0),
                     dataset.Instance(1, {0:.2, 1:.7}, label=-1.0),
                     dataset.Instance(2, {0:.5, 1:.5}, label=1.0),
                     dataset.Instance(3, {0:.7, 1:.7}, label=1.0)]
        inst_dict = dict(zip(range(4), instances))
        self.data = dataset.Dataset(instances=inst_dict)
    
    def test1_SIMPLE_test(self):
        # first, establish our learner
        learner = simple_svm_learner.SimpleLearner([self.data])
        # setup a linear kernel
        learner.params = [svm_parameter(kernel_type = LINEAR)]
        # label two points, one +, one -. thus point 0 should be farthest
        # from the separating hyperplane (line) and 3 should be closest
        learner.label_instances([1,3])
        learner.rebuild_models()
        learner.active_learn(1, batch_size=1)
        assert(2 in learner.labeled_datasets[0].get_instance_ids())
        
if __name__ == '__main__':
    unittest.main()