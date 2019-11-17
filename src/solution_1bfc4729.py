
# coding: utf-8

# In[74]:


import pandas as pd
import json
import numpy as np
from sys import argv

def read_json_file(fileName):
    with open(fileName, 'r') as f:
        return json.load(f)   
    
def find_colours(ip):
    """
    >>> find_colours(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    [1, 4]
    """
    all_colours = list(np.unique(ip))    
    all_colours.remove(0)
    where_colours = [(each_colour, np.where(ip == each_colour)) for each_colour in all_colours]
    if (where_colours[0][1][0] < where_colours[1][1][0]):
        return [where_colours[0][0], where_colours[1][0]]
    else:
        return [where_colours[1][0], where_colours[0][0]]

def colour_in_upper_ip(ip, my_colour):
    ip[0] = [my_colour] * 10
    ip[2] = [my_colour] * 10
    ip[0:5, 0] = [my_colour] * 5
    ip[0:5, 9] = [my_colour] * 5
    return ip
    
def solve(my_ip):
    ip = np.array(my_ip)
    #list keeps the order so we know which colour goes on top
    all_colours = find_colours(ip)
    #remove colours
    for each_colour in all_colours:
        ip[ip == each_colour] = 0 
    #fill in top and bottom pattern
    ip = colour_in_upper_ip(ip, all_colours[0])
    ip = np.flipud(ip)
    ip = colour_in_upper_ip(ip, all_colours[0])
    return np.flipud(ip)
    
 

def main():
    df = read_json_file(argv[1])
    
    for df1 in df['train']:
        print(solve(df1['input']))
        print() 
    for df2 in df['test']:
        print(solve(df1['input']))
        print() 
    
        
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()    
    

