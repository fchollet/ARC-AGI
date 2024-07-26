import os
import json

# Directory containing the JSON files
directory = '/Users/shiro/autoresearch/ARC-AGI/data/training'

# Count the number of JSON files in the directory
json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

print("json_files: ", len(json_files))

# json ファイル内のtest key と train key の値の数をカウント
# Initialize counters
num_train_grids = 0
num_test_grids = 0
# Iterate through each JSON file and count the number of grids in 'train' and 'test'
for json_file in json_files:
    with open(os.path.join(directory, json_file), 'r') as f:
        data = json.load(f)
        num_train_grids += len(data.get('train', []))
        num_test_grids += len(data.get('test', []))

print(f"Total number of train grids: {num_train_grids}")
print(f"Total number of test grids: {num_test_grids}")
