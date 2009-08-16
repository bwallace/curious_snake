import unittest
import os
import sys
import random
import pdb
sys.path.append(os.path.abspath('..'))
import dataset


class TestDataset(unittest.TestCase):
    
    def setUp(self):
        self.path_to_data = os.path.join("..","data","data.txt")
        print "reading in data..."
        self.data = dataset.build_dataset_from_file(self.path_to_data)
        print "success"
         
    def test1_LoadData(self):
        ''' Make sure the data is loaded. '''
        self.assertTrue(self.data.size > 0)

    def test2_AssertUniqueIDs(self):
        ''' Assert that the instances in the dataset all have unique identifiers '''
        ids = self.data.get_instance_ids()
        self.assertEquals(len(ids), len(set(ids)))
        
    def test3_RemoveData(self):
        ''' Make sure we can delete instances from a dataset '''
        # pick some ids at random
        some_instance_ids = random.sample(self.data.get_instance_ids(), 20)
        # remove them
        self.data.remove_instances(some_instance_ids)
        # make sure they're gone.
        for inst_id in self.data.get_instance_ids():
            self.assertTrue(inst_id not in some_instance_ids)

    def test4_UndersampleDataset(self):
        pass
    

        
if __name__ == '__main__':
    unittest.main()