import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_tasks(directory):
    tasks = []
    for file in Path(directory).glob('*.json'):
        with open(file, 'r') as f:
            tasks.append(json.load(f))
    return tasks

def visualize_task(task):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(task['train'][0]['input'])
    axs[0].set_title('Input')
    axs[1].imshow(task['train'][0]['output'])
    axs[1].set_title('Output')
    plt.show()

def simple_model(input_grid):
    # This is a very basic model that just copies the input
    return input_grid

def evaluate_model(model, task):
    correct = 0
    total = len(task['test'])
    for test in task['test']:
        prediction = model(test['input'])
        if np.array_equal(prediction, test['output']):
            correct += 1
    return correct / total

# Main execution
if __name__ == "__main__":
    tasks = load_tasks('/Users/shiro/autoresearch/ARC-AGI/data/training')
    
    for task in tasks[:5]:  # Let's look at the first 5 tasks
        visualize_task(task)
        
        accuracy = evaluate_model(simple_model, task)
        print(f"Model accuracy: {accuracy:.2f}")
        print()

    print("Evaluation complete.")