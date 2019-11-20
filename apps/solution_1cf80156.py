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

a,b,c,d = json_arc_reader('1cf80156.json')
cnp = np.array(c[0])
print(cnp)
res = np.where(cnp>0)
print(res)
col_min = min(res[1])
print(col_min)
col_max = max(res[1])
print(col_max)
row_min = min(res[0])
print(row_min)
row_max = max(res[0])
print(row_max)
    
ok = []
for i in range(row_min,row_max+1):
    for j in range(col_min,col_max+1):
#        ok = np.append(ok,cnp[i][j])
        ok.append(cnp[i][j])

   
ok = np.reshape(ok,(row_max-row_min+1,col_max-col_min+1))
print(type(ok))
print(ok.shape)
print(ok)     
