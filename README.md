# Python Code Solutions for The Abstraction and Reasoning Corpus (ARC)

This repository contains hand written python code for the ARC task data, as well as a browser-based interface to try out the tasks manually which was readily available from the previous repository.

*"The ability to create abstractions from knowledge representations is one of the hallmarks of human intelligence. This is one of the reasons which allowed many of our scientists such as Isaac Newton and Albert Einstein to formulate and invent new concepts. Abstract reasoning has long been used as an example that separates human cognition from artificial intelligence(AI)"*

A complete description of the dataset, its goals, and its underlying logic, can be found in: [The Measure of Intelligence](https://arxiv.org/abs/1911.01547).

## Project Structure
The `repository` consists of three directories and one sub directory within the `src` folder.
- `apps`
- `data`
- `src`
    -  `utils`

## Source Code
The `src` directory consists of hand written python code for three of the tasks namely.
- `1cf80156.json`
- `6150a2bd.json`
- `ce22a75a.json`

Additionally, there is a sub directory `utils` which consists of the helper functions in order for the code to run. It contains a python file `common_utility.py` which consists of functions to read the `input json` file and split the `train-test inputs and outputs`, it also contains another function to visualize the inputs and output after running through the solution files in order to emulate the `testinginterface.html`.  

## Task file format

The `data` directory contains two subdirectories:

- `data/training`: contains the task files for training (400 tasks). Use these to prototype your algorithm or to train your algorithm to acquire ARC-relevant cognitive priors.
- `data/evaluation`: contains the task files for evaluation (400 tasks). Use these to evaluate your final algorithm. To ensure fair evaluation results, do not leak information from the evaluation set into your algorithm (e.g. by looking at the evaluation tasks yourself during development, or by repeatedly modifying an algorithm while using its evaluation score as feedback).

The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:

- `"train"`: demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
- `"test"`: test input/output pairs. It is a list of "pairs" (typically 1 pair).

A "pair" is a dictionary with two fields:

- `"input"`: the input "grid" for the pair.
- `"output"`: the output "grid" for the pair.

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 1x1 and the largest is 30x30.

## Usage of Command Line Arguments
The solutions can be tested using the command line interface by specifying the path to the input json file in either training or evaluation in the data directory .

`solution_1cf80156.py - python solution_1cf80156.py <<path_to_repository>>\ARC\data\<<training or evaluation\json file>>`

`solution_6150a2bd.py - python solution_6150a2bd.py <<path_to_repository>>\ARC\data\<<training or evaluation\json file>>`

`solution_ce22a75a.py - python solution_ce22a75a.py <<path_to_repository>>\ARC\data\<<training or evaluation\json file>>`


# Functionalities from Original ARC Repository
## Usage of the testing interface

The testing interface is located at `apps/testing_interface.html`. Open it in a web browser (Chrome recommended). It will prompt you to select a task JSON file.

After loading a task, you will enter the test space, which looks like this:

![test space](https://arc-benchmark.s3.amazonaws.com/figs/arc_test_space.png)

On the left, you will see the input/output pairs demonstrating the nature of the task. In the middle, you will see the current test input grid. On the right, you will see the controls you can use to construct the corresponding output grid.

You have access to the following tools:

### Grid controls

- Resize: input a grid size (e.g. "10x20" or "4x4") and click "Resize". This preserves existing grid content (in the top left corner).
- Copy from input: copy the input grid to the output grid. This is useful for tasks where the output consists of some modification of the input.
- Reset grid: fill the grid with 0s.

### Symbol controls

- Edit: select a color (symbol) from the color picking bar, then click on a cell to set its color.
- Select: click and drag on either the output grid or the input grid to select cells.
    - After selecting cells on the output grid, you can select a color from the color picking to set the color of the selected cells. This is useful to draw solid rectangles or lines.
    - After selecting cells on either the input grid or the output grid, you can press C to copy their content. After copying, you can select a cell on the output grid and press "V" to paste the copied content. You should select the cell in the top left corner of the zone you want to paste into.
- Floodfill: click on a cell from the output grid to color all connected cells to the selected color. "Connected cells" are contiguous cells with the same color.

### Answer validation

When your output grid is ready, click the green "Submit!" button to check your answer. We do not enforce the 3-trials rule.

After you've obtained the correct answer for the current test input grid, you can switch to the next test input grid for the task using the "Next test input" button (if there is any available; most tasks only have one test input).

When you're done with a task, use the "load task" button to open a new task.
