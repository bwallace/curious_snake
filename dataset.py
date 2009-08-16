'''
    dataset.py
    Byron C Wallace
    Tufts Medical Center
        
    This module contains methods and classes for parsing
    and manipulating datasets.
'''

import pdb
import random 

def build_dataset_from_file(fpath, ids_in_file=False, name=""):
    '''
    Builds and returns a dataset from the file @ the path provided. 
    This assumes the file is in a (sparse!) format amenable to libsvm. E.g.,:
      1 3:.5 4:.8 6:.3
      -1 1:.2 3:.8 6:.4
    Would correspond to two instances, with labels 1 and -1, respectively. 
    The dictionary (sparse) representation of the feature vectors maps dimensions to values. 
    '''
    data = open(fpath, 'r').readlines()
    instances = None
    if ids_in_file:
        instances = [line_to_instance(line) for line in data]
    else:
        instances = [line_to_instance(line, inst_id=i) for i, line in enumerate(data)]
    inst_dict = {}
    for inst in instances:
        inst_dict[inst.id] = inst
    return Dataset(inst_dict, name=name)


def line_to_instance(l, inst_id=None):
    '''
    If id is none **we assume that the ids are in the file**;
    i.e., that the first column of the line is an id (e.g., pubmed
    identifier) and the label is thus at l[1]. If an id is given,
    we assume not id is present in the line.
    '''
    label_index = 0 if inst_id is not None else 1
    l = l.replace("\r\n", "").replace("\n", "")
    l_split = l.split(" ")
    l_split = [l for l in l_split if l != '']
    if inst_id is None:
        inst_id = l_split[0]
    try:
        label = eval(l_split[label_index])
    except:
        print "uh-oh.. problem processing line %s" % l
        pdb.set_trace()
    point = l_split[label_index+1:]
    dict_point = {}
    
    try:
        for coord, value in [dimension.split(":") for dimension in point if point[0] != '']:
          dict_point[eval(coord)] = eval(value)
    except:
        pdb.set_trace()

    return Instance(inst_id, dict_point, label)

        
class Instance:
    '''
    Represents a single point/label combination. The label doesn't necessarily
    need to be provided. The point should be a dictionary mapping coordinates
    (dimensions) to values.
    '''
    def __init__(self, id, point, label=None, name="", is_synthetic=False):
        self.id = id
        self.point = point
        self.label = label
        self.name = name
        
    def set_synthetic_label(self, synth_lbl):
        self.label = synth_lbl

class Dataset:
    '''
    This class represents a set of data. It is comprised mainly of a dictionary mapping
    ids to feature vectors, and various operations -- e.g., undersampling -- can be performed
    on this data.
    '''
    minority_class = 1

    def __len__(self):
        return self.size()

    def __init__(self, instances=None, name=""):
        # instances maps ids to feature vector representations;
        # it needs to be a dictionary.
        self.instances = instances or dict({})
        assert(isinstance(self.instances, dict))
        self.name = name

    def size(self):
        if self.instances is not None:
            return len(self.instances)
        else:
            return 0

    def remove_instances(self, ids_to_remove):
        ''' Remove and return the instances with ids in ids_to_remove '''
        return [self.instances.pop(id) for id in ids_to_remove]

    def copy(self):
        return Dataset(instances = self.instances.copy(), name=self.name) 

    def get_point_for_id(self, id):
        return self.instances[id].point

    def undersample(self, n):
        ''' 
        Remove and return a random subset of n *majority* examples
        from this dataset
        '''
        majority_ids = self.get_list_of_majority_ids()
        print "total number of examples: %s; number of majority examples: %s, number of minority examples: %s" % \
                (len(self.instances), len(self.get_minority_examples()), len(majority_ids))
        picked_so_far = 0

        if len(majority_ids) < n:
            raise Exception, "you asked me to remove more (majority) instances than I have!"
        remove_these = random.sample(majority_ids, n)
        for inst_id in remove_these:
            self.instances.pop(inst_id)         
        return remove_these


    def add_instances(self, instances_to_add):
        '''
        Adds every instance in the instances list to this dataset.
        '''
        for inst in instances_to_add:
            if inst.id in self.instances.keys():
                raise Exception, "dataset.py: error adding instances; duplicate instance ids!"
            self.instances[inst.id] = inst


    def pick_random_minority_instances(self, k):
        min_ids = self.get_list_of_minority_ids()

        if not len(min_ids) >= k:
            raise Exception, "not enough minority examples in dataset!"

        ids = random.sample(min_ids, k)
        return [self.instances[id] for id in ids]


    def pick_random_majority_instances(self, k):
        maj_ids = self.get_list_of_majority_ids()

        if not len(maj_ids) >= k:
            raise Exception, "not enough majority examples in dataset!"

        ids = random.sample(maj_ids, k)
        return [self.instances[id] for id in ids]

    def get_list_of_minority_ids(self, ids_only=True):
        minorities = []
        for id, inst in self.instances.items():
            if inst.label == self.minority_class:
                if ids_only:
                    minorities.append(id)
                else:
                    minorities.append(inst)
        return minorities

    def get_minority_examples(self):
        return self.get_list_of_minority_ids(ids_only=False)

    def get_points_str(self):
        out_s = []
        for inst in self.instances.values():
            inst_str = []
            inst_str.append(str(inst.label))
            for v in inst.point.values():
                inst_str.append(str(v))
            out_s.append(",".join(inst_str))
        return "\n".join(out_s)


    def get_list_of_majority_ids(self, majority_id=-1, ids_only=True):
        majorities = []
        for id, inst in self.instances.items():
            inst_lbl = inst.label
            if inst_lbl == majority_id:
                if ids_only:
                    majorities.append(inst.id)
                else:
                    majorities.append(inst)
        return majorities

    def get_majority_examples(self):
        return self.get_list_of_majority_ids(ids_only=False)

    def number_of_minority_examples(self):
        '''
        Counts and returns the number of minority examples in this dataset.
        '''
        return len(self.get_minority_examples())

    def get_instance_ids(self):
        return self.instances.keys()


    def number_of_majority_examples(self):
        ''' Counts and returns the number of majority examples in this dataset. '''
        return len(self.instances) - self.number_of_minority_examples()


    def get_examples_with_label(self, label):
        ''' Returns a new dataset with all the examples that have the parametric label. '''
        examples = []
        for inst in self.instances.values():
            if inst.label == label:
                examples.append(inst)
        return Dataset(examples)


    def get_and_remove_random_subset(self, n):
        ''' Remove and return a random subset of n examples from this dataset'''
        subset = random.sample(self.instances.keys(), n)
        return self.remove_instances(subset)

    def get_samples(self):
        return [inst.point for inst in self.instances.values()]

    def get_labels(self):
        return [inst.label for inst in self.instances.values()]


    def get_samples_and_labels_for_ids(self, ids):
        samples, labels = [], []
        for id in ids:
            inst = self.instances[id]
            samples.append(inst.point)
            labels.append(inst.label)
        return [samples, labels]

    def get_samples_and_labels(self):
        '''
        Returns a tuple of [[s_1, s_2, ..., s_n], [l_1, l_2, ..., l_n]] where s_i is the ith feature 
        vector and l_i is its label.
        '''
        samples = []
        labels = []
        for inst in self.instances.values():
            samples.append(inst.point)
            labels.append(inst.label)   
        return [samples, labels]
    
  
    