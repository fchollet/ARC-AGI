import json
import numpy as np

def json_arc_reader(file_name):
    json_file = open(file_name)
    data = json.load(json_file)
    print(data)
    train_inputs = [data['train'][i]['input'] for i in range(len(data['train']))]
    train_outputs = [data['train'][i]['output'] for i in range(len(data['train']))]
    test_inputs = [data['test'][i]['input'] for i in range(len(data['test']))]
    test_outputs = [data['test'][i]['output'] for i in range(len(data['test']))]
    return train_inputs,train_outputs,test_inputs,test_outputs
    
a,b,c,d = json_arc_reader('6150a2bd.json')
print(type(d[0]))
cnp = np.array(c[0])
reverse = cnp[::-1]
print(reverse)
print(reverse.shape)
x, y = reverse.shape
rev = []
for i in range(0,3):
        rev_list = reverse[i]
        rev.append(rev_list[::-1])
print(rev)
'''rev = np.hstack(rev).ravel()
print(rev)
rev = np.reshape(rev,(3,3))
print(rev)
print(type(rev))'''
rev = rev.tolist()
print(rev)
print(type(rev))