import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# OpenAI APIキーを環境変数から取得
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Agent:
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger
        self.conversation_history = []

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        self.logger.info(f"{self.name} - {role.capitalize()}:")
        self.logger.info(content)
        self.logger.info("-" * 50)  # セパレータを追加

    def get_response(self, prompt):
        self.add_message("user", prompt)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.conversation_history
        )
        
        assistant_response = response.choices[0].message.content
        self.add_message("assistant", assistant_response)
        
        return assistant_response

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def extract_content(response, tag):
    import re
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        print(f"Warning: <{tag}> tag not found in the response. Returning full response.")
        return response

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

def grid_to_string(grid):
    return '\n'.join([' '.join(map(str, row)) for row in grid])

def verify_prediction(predicted_grid, true_grid):
    return np.array_equal(predicted_grid, true_grid)

# プロンプトテンプレート
system_prompt_template = """You are {agent_name}, an AI assistant specialized in analyzing and predicting patterns in grid-based visual tasks. You are part of a collaborative effort where you and another agent each have access to different parts of the training data for an ARC (Abstraction and Reasoning Corpus) task.

You have access to {data_type} of the training examples, while the other agent has access to the other half. Your goal is to understand the underlying transformation rule and apply it to new test inputs."""

user_prompt_template_a = """
You have access to the following training examples:

{train_examples}

Explanation: Each example consists of an input grid and its corresponding output grid. The numbers represent different colors or states in the grid.
Question: Based on these examples, what do you think is the underlying transformation rule? How would you apply this rule to a new input?
Enclose your reasoning in <Reasoning></Reasoning>. Enclose your proposed transformation rule in <Rule></Rule>.

Previous discussion:
{discussion_history}

{agent_b_instruction}

Consider the information above and provide your updated hypothesis if necessary.
"""

user_prompt_template_b = """
You have access to the following training examples:

{train_examples}

Explanation: Each example consists of an input grid and its corresponding output grid. The numbers represent different colors or states in the grid.
Question: Based on these examples and Agent-A's hypothesis, what do you think is the underlying transformation rule? How would you apply this rule to a new input?
Enclose your reasoning in <Reasoning></Reasoning>. Enclose your proposed transformation rule in <Rule></Rule>.

Previous discussion:
{discussion_history}

Consider the information above, especially Agent-A's latest hypothesis, and provide your updated hypothesis if necessary.
"""

test_prompt_template = """
We have agreed upon the following transformation rule for this ARC task:

{final_rule}

Now, please apply this transformation rule to the following test input:

{test_input}

Provide the predicted output grid. Make sure your prediction follows the agreed-upon rule.
Enclose the output grid in <Output></Output>.
Also, provide a brief explanation of how you applied the rule in <Explanation></Explanation>.
"""

def run_experiment(task, max_turns=2):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"arc_task_log_{timestamp}.log")
    logger = setup_logger('experiment_logger', log_file)

    train_data = task['train']
    test_data = task['test']

    logger.info("Experiment Start")
    logger.info("=" * 50)
    logger.info("=" * 50)

    agent_a = Agent("Agent-A", logger)
    agent_b = Agent("Agent-B", logger)

    # エージェントの初期設定
    agent_a.add_message("system", system_prompt_template.format(agent_name="Agent-A", data_type="the first half"))
    agent_b.add_message("system", system_prompt_template.format(agent_name="Agent-B", data_type="the second half"))

    # 訓練データを分割
    mid = len(train_data) // 2
    train_data_a = train_data[:mid]
    train_data_b = train_data[mid:]

    discussion_history = []
    turn = 0
    final_rule = None

    while turn < max_turns:
        # Agent-A のターン
        train_examples_a = "\n\n".join([f"Input:\n{grid_to_string(ex['input'])}\n\nOutput:\n{grid_to_string(ex['output'])}" for ex in train_data_a])
        agent_b_instruction = "This is the first turn, so there's no previous answer from Agent-B to consider." if turn == 0 else "Consider Agent-B's latest hypothesis."
        agent_a_response = agent_a.get_response(user_prompt_template_a.format(
            train_examples=train_examples_a,
            discussion_history="\n".join(discussion_history),
            agent_b_instruction=agent_b_instruction
        ))
        agent_a_rule = extract_content(agent_a_response, "Rule")
        discussion_history.append(f"Agent-A: {agent_a_rule}")

        # Agent-B のターン
        train_examples_b = "\n\n".join([f"Input:\n{grid_to_string(ex['input'])}\n\nOutput:\n{grid_to_string(ex['output'])}" for ex in train_data_b])
        agent_b_response = agent_b.get_response(user_prompt_template_b.format(
            train_examples=train_examples_b,
            discussion_history="\n".join(discussion_history)
        ))
        agent_b_rule = extract_content(agent_b_response, "Rule")
        discussion_history.append(f"Agent-B: {agent_b_rule}")

        turn += 1

    # 最終的な変換ルールを決定（ここでは単純に最後のAgent-Bの回答を使用）
    final_rule = agent_b_rule

    logger.info("=" * 50)
    logger.info("Final Transformation Rule:")
    logger.info(final_rule)
    logger.info("=" * 50)

    # テストデータに対する予測
    correct_predictions = 0
    total_predictions = len(test_data)

    for test_example in test_data:
        test_input = grid_to_string(test_example['input'])
        test_prompt = test_prompt_template.format(final_rule=final_rule, test_input=test_input)
        
        # Agent-Aに予測させる（Agent-Bでも良いですが、ここではAgent-Aを選択）
        prediction_response = agent_a.get_response(test_prompt)
        predicted_output_str = extract_content(prediction_response, "Output")
        predicted_output = [list(map(int, row.split())) for row in predicted_output_str.split('\n')]

        explanation = extract_content(prediction_response, "Explanation")
        logger.info(f"Test Input:\n{test_input}")
        logger.info(f"Predicted Output:\n{predicted_output_str}")
        logger.info(f"Explanation: {explanation}")

        if verify_prediction(predicted_output, test_example['output']):
            correct_predictions += 1
            logger.info("Prediction: Correct")
        else:
            logger.info("Prediction: Incorrect")
            logger.info(f"True Output:\n{grid_to_string(test_example['output'])}")

        logger.info("-" * 50)

    accuracy = correct_predictions / total_predictions
    logger.info(f"Test Accuracy: {accuracy:.2f}")
    logger.info("=" * 50)

    return accuracy

if __name__ == "__main__":
    tasks = load_tasks('/Users/shiro/autoresearch/ARC-AGI/data/training')
    
    for task in tasks[:5]:  # 最初の5つのタスクを処理
        visualize_task(task)
        
        accuracy = run_experiment(task, max_turns=2)
        print(f"Model accuracy: {accuracy:.2f}")
        print()

    print("Evaluation complete.")