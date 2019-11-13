
# coding: utf-8

# In[16]:


import pandas as pd
import json

def main():
#     content = json.loads('C:/dev/git/ARC/data/training/0a938d79.json')
#     print(content)
#     df = pd.read_json('C:/dev/git/ARC/data/training/0a938d79.json')
#     print(df.head())
    with open('C:/dev/git/ARC/data/training/0a938d79.json', 'r') as f:
        print(f)
        df = json.load(f)
        
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()    
    

