import json as js
import numpy as np

def solve(inputmatrix):
    result=[]
    y = np.array([np.array(xi) for xi in inputmatrix])
    if len(np.unique(y[:1][0]))>1:
        j=(list(np.unique(y[:1][0])))
        indexes = np.unique(y[:1][0], return_index=True)[1]
        k=[y[:1][0][index] for index in sorted(indexes)]
        result.append(k)
    else:
        indexes = np.unique(y[:, 0], return_index=True)[1]
        k = [y[:, 0][index] for index in sorted(indexes)]
        for i in k:
            result.append([i])
    return (result)

with open("D:\\ARC1\\data\\training\\4be741c5.json") as json_file:
    data = js.load(json_file)
inter_data={}
all_data={}
i=1
for input in (data['train']):
    inputmatrix=(input['input'])
    output=solve(inputmatrix)
    inter_data[i]=output
    i+=1

for input in (data['test']):
    inputmatrix=(input['input'])
    output=solve(inputmatrix)
    inter_data['test']=output

all_data['output']=inter_data
print(all_data)

