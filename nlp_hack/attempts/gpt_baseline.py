from nlp_hack.utils.eval_utils import evaluate, get_train_samples, get_test_sample
from nlp_hack.oai import call_openai_api, DEPLOYMENTS
import json


def gpt_baseline(model, task_file):
    prompt_template = """Task Introduction: I will give you a problem that is
    part of a psychometric intelligence test. It is targeted at both humans
    and artificially intelligent systems that aim at emulating a human-like
    form of general fluid intelligence. I would like you to try to solve it
    in the best of your capability, hopefully outperforming human intelligence.

    Description:

    A test-taker is said to solve a task when, upon seeing the task for the
    first time, they are able to produce the correct output grid for the
    test input grid.

    There are a number of task train demonstrations, each consisting of a
    demonstration input/output pair. The strategy for solving the test
    task is to observe the examples and generalize the pattern to the
    test input.
    
    Only produce the output grid for the test input grid. Do not produce any
    other information. The output must be parsable as a 2D Python list.

    <demonstration>
    {demo_json}
    </demonstration>

    <test_input>
    {test_json}
    </test_input>
    """

    train_samples = get_train_samples(task_file)
    test_sample = get_test_sample(task_file)

    train_samples = json.dumps(train_samples, indent=4)
    test_input = json.dumps(test_sample[0]["input"], indent=4)

    response = call_openai_api(
        [
            {
                "role": "user",
                "content": prompt_template.format(
                    demo_json=train_samples, test_json=test_input
                ),
            }
        ],
        DEPLOYMENTS[model],
    )

    return response


if __name__ == "__main__":
    # TODO: Iterate over files and pass to evaluate
    print(gpt_baseline("gpt-35", "../../data/training/0a938d79.json"))
