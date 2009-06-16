'''
    dataset.py
    Byron C Wallace
    Tufts Medical Center
        
    This module contains methods and classes for parsing
    and manipulating datasets.
'''

import pdb
import random 

def build_dataset_from_file(fpath):
    '''
    Builds and returns a dataset from the file @ the path provided. 
    This assumes the file is in a (sparse!) format amenable to libsvm. E.g.,:
      1 3:.5 4:.8 6:.3
      -1 1:.2 3:.8 6:.4
    Would correspond to two instances, with labels 1 and -1, respectively. 
    The dictionary (sparse) representation of the feature vectors maps dimensions to values. 
  
    An instance number is assigned to each example in the dataset. 
    These instance numbers correspond to the (zero-based) order of the points in the file 
    (the instance on the first line will get id 0; the instance on the second, 1; and so on).  
    '''
    data = open(fpath, 'r').readlines()
    instances = [line_to_instance(data[id], id) for id in range(len(data))]
    return dataset(instances)

def line_to_instance(l, id):
    l = l.replace("\r\n", "")
    l = l.replace("\n", "")
    l_split = l.split(" ")
    l_split = [l for l in l_split if l != '']
    try:
        label = eval(l_split[0])
    except:
        pdb.set_trace()
    point = l_split[1:]
    dict_point = {}
    
    try:
        for coord, value in [dimension.split(":") for dimension in point if point[0] != '']:
          dict_point[eval(coord)] = eval(value)
    except:
        pdb.set_trace()


    return instance(id, dict_point, label)

        
class instance:
    '''
    Represents a single point/label combination. The label doesn't necessarily
    need to be provided. The point should be a dictionary mapping coordinates
    (dimensions) to values.
    '''
    point = None
    label = None
    id = None
    name = None

  
    def __init__(self, id, point, label=None, name="", is_synthetic=False):
        self.id = id
        self.point = point
        self.label = label
        self.name = name
        
    def set_synthetic_label(self, synth_lbl):
        self.label = synth_lbl

class dataset:
    '''
    This class represents a set of data. It is comprised mainly of a list of instances, and 
    various operations -- e.g., undersampling -- can be performed on this list.
    '''
    instances = None
    minority_class = 1
    name = None
    
    def __init__(self, instances, name=""):
        self.instances = instances
        self.name = None
    
    def size(self):
        if self.instances is not None:
            return len(self.instances)
        else:
            return None
        
    def remove_instances(self, ids_to_remove):
        '''
        Remove and return the instances with ids found in the
        parametric list.
        '''
        removed_instances = []
        for instance_id in ids_to_remove:
            for instance in self.instances:
              if (instance.id == instance_id):
                removed_instances.append(instance)
                self.instances.remove(instance)
                break
        return removed_instances
      
       
    def copy(self):
         return dataset(list(self.instances)) 
      
      
    def undersample(self, n):
        ''' 
        Remove and return a random subset of n *majority* examples
         from this dataset
         '''
        print "total number of examples: %s; number of majority examples: %s, number of minority examples: %s" % (len(self.instances), len(self.get_minority_examples().instances), len(self.get_majority_examples().instances))
        picked_so_far = 0
        majorities = self.get_majority_examples().instances
        if len(majorities) < n:
            pdb.set_trace()
        picked = random.sample(majorities, n)
        for cur_pick in picked:
            self.instances.remove(cur_pick)         
        return picked
    
  
    def add_instances(self, instances):
        '''
        Adds every instance in the instances list to this dataset.
        '''
        for inst in instances:
            self.instances.append(inst)
        # ascertain that we have no duplicate ids
        self.assert_unique_instances()
        
    def assert_unique_instances(self):
        ids = self.get_instance_ids()
        if not len(ids) == len(set(ids)):
            pdb.set_trace()
            raise Exception, "duplicate instance ids!"
            
         
    def pick_random_minority_instances(self, k):
        min_ids = self.get_list_of_minority_ids()

        if not len(min_ids) >= k:
            raise Exception, "not enough minority examples in dataset!"
            
        selected_ids = []
        while len(selected_ids) < k:
            selected_id = random.choice(min_ids)
            min_ids.remove(selected_id)
            selected_ids.append(selected_id)
        
        return selected_ids
        
        
    def pick_random_majority_instances(self, k):
        maj_ids = self.get_list_of_majority_ids()
 
        if not len(maj_ids) >= k:
            raise Exception, "not enough majority examples in dataset!"

        selected_ids = []
        while len(selected_ids) < k:
            selected_id = random.choice(maj_ids)
            maj_ids.remove(selected_id)
            selected_ids.append(selected_id)
        
        return selected_ids
        
    def get_list_of_minority_ids(self, include_false_minorities=False):
        minorities = []
        for inst in self.instances:
            inst_lbl = inst.label
            if include_false_minorities:
                inst_lbl = inst.label
            if inst_lbl == self.minority_class:
                minorities.append(inst.id)
        return minorities
        
    def get_minority_examples(self):
        minorities = []
        for inst in self.instances:
            if inst.label == self.minority_class:
                minorities.append(inst)
        return dataset(minorities)
        
    def get_points_str(self):
        out_s = []
        for inst in self.instances:
            inst_str = []
            inst_str.append(str(inst.label))
            for v in inst.point.values():
                inst_str.append(str(v))
            out_s.append(",".join(inst_str))
        return "\n".join(out_s)
        
        
    def get_list_of_majority_ids(self, majority_id=-1):
        majorities = []
        for inst in self.instances:
            inst_lbl = inst.label
            if inst_lbl == majority_id:
                majorities.append(inst.id)
        return majorities
        
    def number_of_minority_examples(self):
        '''
        Counts and returns the number of minority examples in this dataset.
        '''
        return len([inst for inst in self.instances if inst.label == self.minority_class])
    
    
    def get_instance_ids(self):
        return [inst.id for inst in self.instances]
    
    def get_majority_examples(self):
        majorities = []
        for inst in self.instances:
            if inst.label != self.minority_class:
                majorities.append(inst)
        return dataset(majorities)

    def number_of_majority_examples(self):
        '''
        Counts and returns the number of majority examples in this dataset.
        '''
        return len(self.instances) - self.number_of_minority_examples()
    
    
    def get_examples_with_label(self, label):
        '''
        Returns a new dataset with all the examples that have the parametric label.
        '''
        examples = []
        for inst in self.instances:
            if inst.label == label:
                examples.append(inst)
        return dataset(examples)
        
    
    def get_and_remove_random_subset(self, n):
        '''
        Remove and return a random subset of n examples from this 
        dataset
        '''
        subset = []
        for i in range(n):
            cur_pick = random.choice(self.instances)
            subset.append(cur_pick)
            self.instances.remove(cur_pick)
        return subset
    
    def get_random_subset(self, n):
        subset = []
        instances_copy = list(self.instances)
        for i in range(n):
            cur_pick = random.choice(instances_copy)
            subset.append(cur_pick)
            instances_copy.remove(cur_pick)
        return subset
        
            
    def get_samples(self):
        return [inst.point for inst in self.instances]

    def get_labels(self):
        return [inst.label for inst in self.instances]

    def get_samples_and_labels(self):
        '''
        Returns a tuple of [[s_1, s_2, ..., s_n], [l_1, l_2, ..., l_n]] where s_i is the ith feature 
        vector and l_i is its label.
        
        Note that by default this returns the (possibly fake) label attribute, not the "real_label"
        '''
        samples = []
        labels = []
        for inst in self.instances:
            samples.append(inst.point)
            labels.append(inst.label)   
        return [samples, labels]
    
  
    