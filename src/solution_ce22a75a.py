import sys
import numpy as np
import utils.read_json as rj


with open(sys.argv[1], 'r') as j_file:
    a,b,c,d =rj.json_arc_reader(j_file)
    test_result = np.array(c[0])
    res = np.where(test_result>0)
    for i in list(zip(res[0], res[1])):     
        for x in range(-1,2):
          for y in range(-1,2):
            if (x != 0 or y != 0):
                test_result[i[0]-x][i[1]-y] = 1
                test_result[i[0]][i[1]] = 1            

print(test_result)
plt_list = [c[0], test_result]
rj.visualize(plt_list)