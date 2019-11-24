import sys
import numpy as np
import utils.read_json as rj

with open(sys.argv[1], 'r') as j_file:
    a,b,c,d = rj.json_arc_reader(j_file)
    cnp = np.array(c[0])
    res = np.where(cnp>0)
    col_min = min(res[1])
    col_max = max(res[1])
    row_min = min(res[0])
    row_max = max(res[0])

test_result = []
for i in range(row_min,row_max+1):
    for j in range(col_min,col_max+1):
        test_result.append(cnp[i][j])

test_result = np.reshape(test_result,(row_max-row_min+1,col_max-col_min+1))

print(test_result)
plt_list = [c[0], test_result]
rj.visualize(plt_list)
