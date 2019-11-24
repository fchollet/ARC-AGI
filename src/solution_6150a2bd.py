import sys
import utils.read_json as rj
import numpy as np

with open(sys.argv[1], 'r') as j_file:
    a,b,c,d =rj.json_arc_reader(j_file)  
    cnp = np.array(c[0])
    test_result = np.flip(cnp)
    print(test_result)

print(test_result)
plt_list = [c[0], test_result]
rj.visualize(plt_list)
