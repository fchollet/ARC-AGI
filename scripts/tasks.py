from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from beartype import beartype
from utils import find_islands, get_element_with_highest_frequency


class Task(ABC):
    NUM_COLORS: int = 10
    EMPTY_COLOR = 0
    SEPERATOR_COLOR = 5

    @beartype
    def run(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [self.gen_io() for _ in range(n)]

    @beartype
    @abstractmethod
    def gen_io(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @beartype
    @abstractmethod
    def solve(self, input: np.ndarray) -> np.ndarray:
        pass


class Task1(Task):
    def __init__(self, max_canvas_size: int = 10):
        self.base_color = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.base_color})
        self.fill_color = np.random.choice(remaining_colors)
        self.max_canvas_size = max_canvas_size

    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = self.gen_input(canvas_size)
        output = self.solve(input)

        return input, output

    def gen_input(self, canvas_size: int) -> np.ndarray:
        input = np.random.choice([0, self.base_color], size=(canvas_size, canvas_size), p=[0.6, 0.4])
        input = np.pad(input, ((1, 1), (1, 1)), constant_values=self.EMPTY_COLOR)
        if not self.check_is_valid(input):
            if np.random.rand() < 0.90:
                input = self.insert_rand_box(input, canvas_size)
        return input

    def insert_rand_box(self, input: np.ndarray, canvas_size: int) -> np.ndarray:
        # num_boxes = np.random.randint(1, 1 + (canvas_size - 3) // 2)
        w = np.random.randint(3, canvas_size + 1)
        h = np.random.randint(3, canvas_size + 1)
        x = np.random.randint(1, canvas_size - w + 2)
        y = np.random.randint(1, canvas_size - h + 2)
        input[x + 1 : x + w - 1, y + 1 : y + h - 1] = self.EMPTY_COLOR
        input[x, y : y + h] = self.base_color
        input[x + w - 1, y : y + h] = self.base_color
        input[x : x + w, y] = self.base_color
        input[x : x + w, y + h - 1] = self.base_color
        return input

    def check_is_valid(self, input: np.ndarray) -> bool:
        return len(find_islands(input)) > 0

    def solve(self, input: np.ndarray) -> np.ndarray:
        islands = find_islands(input)
        if len(find_islands(input)) == 0:
            return input
        idx = tuple(map(list, zip(*sum(islands, []))))
        output = np.copy(input)
        output[idx[0], idx[1]] = self.fill_color
        return output


class Task2(Task):
    def __init__(self, max_canvas_size: int = 10):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.input_color})
        self.output_color = np.random.choice(remaining_colors)
        self.max_canvas_size = max_canvas_size

    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = np.random.choice([0, self.input_color], size=(canvas_size, canvas_size))
        output = self.solve(input)

        return input, output

    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.copy(input)
        output[output == self.input_color] = self.output_color
        return output


class Task5(Task):
    def __init__(self, max_canvas_size: int = 5):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(
            set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.SEPERATOR_COLOR, self.input_color}
        )
        self.output_color = np.random.choice(remaining_colors)
        self.max_canvas_size = max_canvas_size

    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = np.random.choice([0, self.input_color], size=(canvas_size, 2 * canvas_size + 1))
        input[:, canvas_size] = self.SEPERATOR_COLOR
        output = self.solve(input)
        return input, output

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        canvas_size = input.shape[0]
        left_input, right_input = input[:, :canvas_size], input[:, canvas_size + 1 :]
        output = left_input & right_input
        output[output == self.input_color] = self.output_color
        return output


class Task372(Task):
    def __init__(self, max_canvas_size: int = 10):
        self.color_1 = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.color_1})
        self.color_2 = np.random.choice(remaining_colors)
        self.max_canvas_size = max_canvas_size

    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = self.gen_input(canvas_size)
        output = self.solve(input)
        return input, output

    def gen_input(self, canvas_size: int) -> np.ndarray:
        input = np.full((2, canvas_size), self.color_1)
        input[1, :] = self.color_2
        return input

    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.copy(input)
        output[0, 1::2] = self.color_2
        output[1, 1::2] = self.color_1
        return output


class Taskx1(Task):
    def __init__(self, max_canvas_size: int = 10):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        self.max_canvas_size = max_canvas_size

    @beartype
    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = np.random.choice([0, self.input_color], size=(canvas_size, canvas_size))
        output = self.solve(input)
        return input, output

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.where(input == self.input_color, self.EMPTY_COLOR, self.input_color)
        return output


class Taskx2(Task):
    def __init__(self, max_canvas_size: int = 10):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.input_color})
        self.output_color = np.random.choice(remaining_colors)
        self.max_canvas_size = max_canvas_size

    @beartype
    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = np.random.choice([0, self.input_color], size=(canvas_size, canvas_size))
        output = self.solve(input)
        return input, output

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.where(input == self.input_color, self.EMPTY_COLOR, self.output_color)
        return output


class Taskx3(Task):
    def __init__(self, max_canvas_size: int = 10):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        self.max_canvas_size = max_canvas_size

    @beartype
    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = np.random.choice([0, self.input_color], size=(canvas_size, canvas_size), p=[0.7, 0.3])
        output = self.solve(input)
        return input, output

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output_color = self.input_color if np.any(input == self.input_color) else self.EMPTY_COLOR
        output = np.full_like(input, output_color)
        return output


class Taskx4(Task):
    def __init__(self, max_canvas_size: int = 5):
        self.input_colors = range(1, self.NUM_COLORS)
        self.max_canvas_size = max_canvas_size

    @beartype
    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = self.gen_input(canvas_size)
        output = self.solve(input)
        return input, output

    @beartype
    def gen_input(self, canvas_size: int) -> np.ndarray:
        input = np.random.choice(self.input_colors, size=(canvas_size, canvas_size))
        if not self.check_if_valid(input):
            unique, counts = np.unique(input, return_counts=True)
            min_count_color = unique[np.argmin(counts)]
            max_count_color = unique[np.random.choice(np.flatnonzero(counts == counts.max()))]
            input[input == min_count_color] = max_count_color

        return input

    @beartype
    def check_if_valid(self, input: np.ndarray) -> bool:
        hf_elem = get_element_with_highest_frequency(input, is_unique=True)
        return hf_elem is not None

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        out_color = get_element_with_highest_frequency(input)
        output = np.full_like(input, out_color)
        return output


class Taskx5(Task):
    def __init__(self, max_canvas_size: int = 10):
        self.color_1 = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.color_1})
        self.color_2 = np.random.choice(remaining_colors)
        self.max_canvas_size = max_canvas_size

    def gen_io(self) -> Tuple[np.ndarray, np.ndarray]:
        canvas_size = np.random.randint(3, self.max_canvas_size)
        input = self.gen_input(canvas_size)
        output = self.solve(input)
        return input, output

    def gen_input(self, canvas_size: int) -> np.ndarray:
        input = np.full((canvas_size, 2), self.color_1)
        input[:, 1] = self.color_2
        return input

    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.copy(input)
        output[1::2, 0] = self.color_2
        output[1::2, 1] = self.color_1
        return output
