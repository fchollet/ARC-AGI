import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 

def json_arc_reader(file_name):
    json_file = open(file_name)
    data = json.load(json_file)
    #print(data)
    train_inputs = [data['train'][i]['input'] for i in range(len(data['train']))]
    train_outputs = [data['train'][i]['output'] for i in range(len(data['train']))]
    test_inputs = [data['test'][i]['input'] for i in range(len(data['test']))]
    test_outputs = [data['test'][i]['output'] for i in range(len(data['test']))]
    return train_inputs,train_outputs,test_inputs,test_outputs
#Added pathlib to set path irrespective of the os for file load operations
data_folder = Path("../data/training/")
file_path = data_folder / "1cf80156.json"
a,b,c,d = json_arc_reader(file_path)
cnp = np.array(c[0])
res = np.where(cnp>0)
col_min = min(res[1])
col_max = max(res[1])
row_min = min(res[0])
row_max = max(res[0])
    
ok = []
for i in range(row_min,row_max+1):
    for j in range(col_min,col_max+1):
        ok.append(cnp[i][j])

   
ok = np.reshape(ok,(row_max-row_min+1,col_max-col_min+1))

print(ok)    
print(d[0])

plt.matshow(c[0])
plt.show()  

plt.matshow(ok)
plt.show()   
