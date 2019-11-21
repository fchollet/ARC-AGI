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

a,b,c,d = json_arc_reader('ce22a75a.json')
cnp = np.array(c[0])
res = np.where(cnp>0)
for i in list(zip(res[0], res[1])):     
    for x in range(-1,2):
      for y in range(-1,2):
        if (x != 0 or y != 0):
            cnp[i[0]-x][i[1]-y] = 1
            cnp[i[0]][i[1]] = 1
    
    
#    cnp[i[0]-1][i[1]-1] = 1
#    cnp[i[0]][i[1]-1] = 1
#    cnp[i[0]+1][i[1]-1] = 1
#    
#    cnp[i[0]-1][i[1]] = 1
#    cnp[i[0]+1][i[1]] = 1
#    
#    cnp[i[0]-1][i[1]+1] = 1
#    cnp[i[0]][i[1]+1] = 1
#    cnp[i[0]+1][i[1]+1] = 1

