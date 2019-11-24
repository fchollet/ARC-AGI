import json
import matplotlib.pyplot as plt

def json_arc_reader(json_file):
    
    data = json.load(json_file)
    
    train_inputs = [data['train'][i]['input'] for i in range(len(data['train']))]
    train_outputs = [data['train'][i]['output'] for i in range(len(data['train']))]
    test_inputs = [data['test'][i]['input'] for i in range(len(data['test']))]
    test_outputs = [data['test'][i]['output'] for i in range(len(data['test']))]
    return train_inputs,train_outputs,test_inputs,test_outputs


def visualize(input):
    for i in range(len(input)):
        plt.matshow(input[i])
        plt.show()