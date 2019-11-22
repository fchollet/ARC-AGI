import sys
import numpy as np
import Read_json


with open(sys.argv[1], 'r') as j_file:
    a,b,c,d =Read_json.json_arc_reader(j_file)
    cnp = np.array(c[0])
    res = np.where(cnp>0)
    for i in list(zip(res[0], res[1])):     
        for x in range(-1,2):
          for y in range(-1,2):
            if (x != 0 or y != 0):
                cnp[i[0]-x][i[1]-y] = 1
                cnp[i[0]][i[1]] = 1            
    print(cnp)