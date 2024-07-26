import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI
import os
import logging
from datetime import datetime
import traceback

# Set up logging
log_directory = Path("logs")
log_directory.mkdir(exist_ok=True)
log_filename = f"arc_solver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = log_directory / log_filename

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()  # This will also output logs to console
    ]
)
logger = logging.getLogger(__name__)

# Global prompt template
PROMPT_TEMPLATE = """
You will be given multiple paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. Your task is to determine the transformation rule and implement it in code.
The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). Each number corresponds to a color in the image as follows: 0: Black, 1: Blue, 2: Red, 3: Green, 4: Yellow, 5: Grey, 6: Pink, 7: Orange, 8: Purple, 9: Brown.

Here are the training examples:
{train_examples}

The transformation rule should be unambiguous and applicable to all the provided example inputs and outputs. It should also work for new, unseen inputs with similar patterns.

You'll need to carefully reason to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

After your reasoning, write code in triple backticks (`python and then` ). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).

Don't write tests in your python code, just output the `transform` function. (It will be tested later.)

You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion.

Your reasoning can be as long as necessary. The goal is to ensure you end up with a correct implementation of the transformation rule.

You are creative and accomplished at solving puzzles. Please determine the transformation rule and implement it in the `transform` function.
"""

CORRECTION_PROMPT = """
The transform function you provided does not correctly transform all the training examples. Please review the following examples and modify the transform function to correctly handle all cases:

{failed_examples}

Please provide an updated transform function that correctly handles all these cases.
"""

MAX_ITERATIONS = 3

def load_tasks(directory):
    tasks = []
    for file in Path(directory).glob('*.json'):
        with open(file, 'r') as f:
            tasks.append(json.load(f))
    logger.info(f"Loaded {len(tasks)} tasks from {directory}")
    return tasks

def grid_to_string(grid):
    return '\n'.join([' '.join(map(str, row)) for row in grid])

def string_to_grid(s):
    return [list(map(int, row.split())) for row in s.strip().split('\n')]

def gpt4_model(task):
    logger.info("Starting GPT-4 model inference")
    
    # Prepare the prompt with all training examples
    train_examples = ""
    for i, example in enumerate(task['train']):
        train_examples += f"\nTraining Example {i+1}:\nInput:\n{grid_to_string(example['input'])}\nOutput:\n{grid_to_string(example['output'])}\n"

    # Use the global prompt template
    prompt = PROMPT_TEMPLATE.format(train_examples=train_examples)

    for iteration in range(MAX_ITERATIONS):
        # Call the GPT-4 API
        try:
            logger.info(f"Sending request to GPT-4 API (iteration {iteration + 1})")
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in solving ARC tasks."},
                    {"role": "user", "content": prompt}
                ]
            )
            logger.info("Received response from GPT-4 API")
        except Exception as e:
            logger.error(f"Error calling GPT-4 API: {str(e)}")
            raise
        
        logger.debug(f"GPT-4 full response: {response.choices[0].message.content}")

        # Extract the Python code from the response
        try:
            code = response.choices[0].message.content.split("```python")[1].split("```")[0]
            logger.debug(f"Extracted code from GPT-4 response:\n{code}")
        except IndexError:
            logger.error("Failed to extract code from GPT-4 response")
            raise ValueError("Unexpected response format from GPT-4")

        # Execute the code to get the transform function
        try:
            local_vars = {}
            exec(code, globals(), local_vars)
            if 'transform' not in local_vars:
                raise ValueError("The generated code does not contain a 'transform' function")
            transform_func = local_vars['transform']
            logger.info("Successfully extracted 'transform' function from GPT-4 generated code")
        except Exception as e:
            logger.error(f"Error executing GPT-4 generated code: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # Validate the transform function on training examples
        failed_examples = []
        for i, example in enumerate(task['train']):
            try:
                prediction = transform_func(example['input'])
                logger.debug(f"Example {i+1} input:\n{grid_to_string(example['input'])}")
                logger.debug(f"Example {i+1} expected output:\n{grid_to_string(example['output'])}")
                logger.debug(f"Example {i+1} actual output:\n{grid_to_string(prediction)}")
                if not np.array_equal(prediction, example['output']):
                    failed_examples.append(f"Example {i+1}:\nInput:\n{grid_to_string(example['input'])}\nExpected Output:\n{grid_to_string(example['output'])}\nActual Output:\n{grid_to_string(prediction)}")
            except Exception as e:
                logger.error(f"Error applying transform function to example {i+1}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                failed_examples.append(f"Example {i+1}: Error - {str(e)}")

        if not failed_examples:
            logger.info("Transform function successfully validated on all training examples")
            return transform_func
        
        logger.warning(f"Transform function failed on {len(failed_examples)} out of {len(task['train'])} examples.")
        
        if iteration == MAX_ITERATIONS - 1:
            logger.warning("Max iterations reached. Using the last generated transform function.")
            return transform_func

        # Prepare correction prompt
        failed_examples_str = "\n\n".join(failed_examples)
        prompt = CORRECTION_PROMPT.format(failed_examples=failed_examples_str)
        logger.info(f"Requesting correction for failed examples.")

    # This line should never be reached due to the return statements above
    raise ValueError("Unexpected error in gpt4_model function")

def evaluate_model(model, task, task_id):
    correct = 0
    total = len(task['test'])
    logger.info(f"Evaluating model on task {task_id}")

    # Get the transform function for this task
    transform_func = model(task)

    for i, test in enumerate(task['test']):
        try:
            prediction = transform_func(test['input'])
            if np.array_equal(prediction, test['output']):
                correct += 1
                logger.info(f"Test case {i+1}/{total} correct")
            else:
                logger.info(f"Test case {i+1}/{total} incorrect")
        except Exception as e:
            logger.error(f"Error in test case {i+1}/{total}: {str(e)}")
    accuracy = correct / total
    logger.info(f"Model accuracy for task {task_id}: {accuracy:.2f}")
    return accuracy

# Main execution
if __name__ == "__main__":
    logger.info("Starting ARC evaluation")
    tasks = load_tasks('/Users/shiro/autoresearch/ARC-AGI/data/training')
    task_num = 3
    
    for i, task in enumerate(tasks[:task_num]):  # Let's look at the first 3 tasks
        logger.info(f"Processing task {i+1}/{task_num}")
        
        try:
            accuracy = evaluate_model(gpt4_model, task, i+1)
            print(f"Model accuracy: {accuracy:.2f}")
            print()
        except Exception as e:
            logger.error(f"Error evaluating task {i+1}: {str(e)}")

    logger.info("Evaluation complete")
    print(f"Evaluation complete. Log file saved at: {log_path}")