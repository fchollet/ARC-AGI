
# coding: utf-8

# In[68]:


import pandas as pd
import json
import numpy as np


def read_json_file(fileName):
    with open(fileName, 'r') as f:
        return json.load(f)   
    
def will_be_vertical_line(ip_array, rows):
#     if (ip[0] == 0 or ip2[0])
    print(ip_array[7][5])
#     print(ip_array.shape)
#     print(ip)
    
def find_details(df_io):
    """
    >>> find(pd.DataFrame({'input': [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'output': [[0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0]]}))
    1
    """ 
    ip = np.array(df_io['input'])
    (rows, cols) = (np.nonzero(ip))
    will_be_vertical_line(ip, ip1, ip2)
    print(df_io['input'])
    print(ip)
    print(ip.shape)
    print(ip1, ip2)
    return 1

def solve(df_io):
#     """
#     >>> calc_total_error( pd.DataFrame({'Name':[12, 14, 16, 12, 17, 11], 'Age':[20, 21, 19, 18, 21, 22], \
#         'variety':['A1', 'B1', 'C1', 'A1', 'A1', 'C1']}), 'variety', ('Age', (21.5, (0.4666666666666666, 'A1', 'C1'))))
#     0.3333333333333333
#     """    
    find_details(df_io)

def main():
#     content = json.loads('C:/dev/git/ARC/data/training/0a938d79.json')
#     print(content)
#     df = pd.read_json('C:/dev/git/ARC/data/training/0a938d79.json')
#     print(df.head())
    df = read_json_file('C:/dev/git/ARC/data/training/0a938d79.json')
#     for df_io in df['train']:
#         print(df_io)
    solve(df['train'][0])
    
    
    
#     for input_output in df['train']:
#         print(input_output)
    
        
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()    
    

