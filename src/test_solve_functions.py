import unittest
import json

import solution_5c0a986e
import solution_b91ae062
import solution_2dc579da
#import solution_4be741c5 # These two solutions need main() functions
#import solution_4c4377d9

# Source https://docs.python.org/3/library/unittest.html
#        [Accessed: 16/11/2019]

class TestSolveFunctions(unittest.TestCase):

    @staticmethod
    def load_json_data(file_name):
        with open(file_name) as f:
            text = f.read()
    
        # Convert from JSON to Python Dictionary
        return json.loads(text)
        
    def test_5c0a986(self):
        file_name = '../data/training/5c0a986e.json'
        data = self.load_json_data(file_name)
        
        for data_train in data['train']:
            solution = solution_5c0a986e.solve(data_train['input'])
            self.assertEqual(solution, data_train['output'])
        
        for data_test in data['test']:
            solution = solution_5c0a986e.solve(data_test['input'])
            self.assertEqual(solution, data_test['output'])
    
    def test_b91ae062(self):
        file_name = '../data/training/b91ae062.json'
        data = self.load_json_data(file_name)
        
        for data_train in data['train']:
            solution = solution_b91ae062.solve(data_train['input'])
            self.assertEqual(solution, data_train['output'])
        
        for data_test in data['test']:
            solution = solution_b91ae062.solve(data_test['input'])
            self.assertEqual(solution, data_test['output'])
            
    def test_2dc579da(self):
        file_name = '../data/training/2dc579da.json'
        data = self.load_json_data(file_name)
        
        for data_train in data['train']:
            solution = solution_2dc579da.solve(data_train['input'])
            self.assertEqual(solution, data_train['output'])
        
        for data_test in data['test']:
            solution = solution_2dc579da.solve(data_test['input'])
            self.assertEqual(solution, data_test['output'])
     
    # Invalid path in the following 2 solutions so tests don't run
#    def test_4be741c5(self):
#        file_name = '../data/training/4be741c5.json'
#        data = self.load_json_data(file_name)
#        
#        for data_train in data['train']:
#            solution = solution_4be741c5.solve(data_train['input'])
#            self.assertEqual(solution, data_train['output'])
#        
#        for data_test in data['test']:
#            solution = solution_4be741c5.solve(data_test['input'])
#            self.assertEqual(solution, data_test['output'])
#            
#    def test_4c4377d9(self):
#        file_name = '../data/training/4c4377d9.json'
#        data = self.load_json_data(file_name)
#        
#        for data_train in data['train']:
#            solution = solution_4c4377d9.solve(data_train['input'])
#            self.assertEqual(solution, data_train['output'])
#        
#        for data_test in data['test']:
#            solution = solution_4c4377d9.solve(data_test['input'])
#            self.assertEqual(solution, data_test['output'])


if __name__ == '__main__':
    unittest.main()
