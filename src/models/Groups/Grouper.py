"""
This class is the base class for all grouping algorithms

With the data that one of the sub classes of this class produces one should be able to layer groups over a frame
and display that to the user

To do this, this class will create several frames relating to each type of possible group.
Each of those frames will be identical to the original frame except it's features will be grouped and one hot encoded

so if a frame had two colors in it scattered about. That frame would then be one hot encoded into two new frames
defined by the color grouper. Each of the squares with a given color on it would be one hot encoded to represent that
group. Then when doing processing later we will be able to do it one group at a time to see if it is relevant.

"""
import numpy as np
from .Group import Group
from collections import deque


class Grouper:

    def __init__(self, problem):
        self.problem = np.array(problem, np.int32)
        self.shape_groups, self.color_groups, self.metadata = self.generate_groups()

    def generate_groups(self):
        """
        This will load the problem into a numpy array and then instantiate several group objects with it
        :return: [frame]
        """
        # two arrays that are of equal length that together are the x and y values of non zero values in the problem
        non_zeros = self.problem.nonzero()
        shape_groups = self.make_groups(non_zeros, color_match=False)
        color_groups = self.make_groups(non_zeros, color_match=True)
        meta_data = {'color': [0 for i in range(10)], 'most_common_color': 1, 'least_common_color': 1}
        for group in color_groups:
            meta_data['color'][group.color] += 1
        most_common_color_occurences = 0
        least_common_color_occurences = 9999
        for group in color_groups:
            if meta_data['color'][group.color] > most_common_color_occurences:
                most_common_color_occurences = meta_data['color'][group.color]
                meta_data['most_common_color'] = group.color
            if meta_data['color'][group.color] < least_common_color_occurences:
                least_common_color_occurences = meta_data['color'][group.color]
                meta_data['least_common_color'] = group.color

        return shape_groups, color_groups, meta_data

    def x_y_color_generator(self, non_zeros):
        for index in range(len(non_zeros[0])):
            yield non_zeros[0][index], non_zeros[1][index], self.problem[non_zeros[0][index]][non_zeros[1][index]]

    def make_groups(self, non_zeros, color_match):
        """

        TODO if we are doing color match then check the color of the group if this color is different, skip it
        TODO if we aren't doing color match, and the group still has a single color when we are done with it. Remove it
        we will handle color coded groups in the color groups
        :param non_zeros:
        :param color_match:
        :return:
        """
        done_dict = {}
        groups = []
        # This outer for loop generates the individual groups based on shape
        for x, y, color in self.x_y_color_generator(non_zeros):
            done_str = str(x) + ',' + str(y)
            if done_str not in done_dict:
                group = Group(x, y, color)
                done_dict[done_str] = color
                deq = self.get_neighboring_non_zero_indices(x, y, color, done_dict, color_match)
                while True:
                    try:
                        deq_x, deq_y, deq_color = deq.pop()
                        group.add_new_cell(deq_x, deq_y, deq_color)
                        done_dict[self.get_done_str(deq_x, deq_y)] = deq_color
                        deq += self.get_neighboring_non_zero_indices(deq_x, deq_y, deq_color, done_dict, color_match)
                    except IndexError:
                        break
                if color_match or (not color_match and group.color == -1):
                    # if we are not color matching then we should only add this shape if it has a non-uniform color
                    # otherwise we will end up with redundant objects
                    groups.append(group)
        return groups

    def get_neighboring_non_zero_indices(self, x, y, color, done_dict, color_match):
        """

        :param color_match:
        :param color:
        :param x:
        :param y:
        :param done_dict:
        :return: deque of new indicies that are valid and should be added to the group
        """
        to_be_dequed = []
        for x_index in range(x-1, x+2):
            for y_index in range(y-1, y+2):
                if x_index < 0 or x_index > len(self.problem) - 1:
                    break
                if x_index == x == y_index == y or y_index < 0 or y_index > len(self.problem[0]) - 1:
                    # We are looking for the neighbors of this occurence. We should skip this
                    continue
                done_str = self.get_done_str(x_index, y_index)
                if done_str not in done_dict:
                    new_color = self.problem[x_index][y_index]
                    if new_color != 0 and (not color_match or (color_match and color == new_color)):
                        to_be_dequed.append((x_index, y_index, new_color))
        return deque(to_be_dequed)

    @staticmethod
    def get_done_str(x, y):
        return str(x) + ',' + str(y)

    def remove_group(self, group):
        for index in range(len(self.color_groups)):
            if group.equals(self.color_groups[index]):
                del self.color_groups[index]
                break
        for index in range(len(self.shape_groups)):
            if group.equals(self.shape_groups[index]):
                del self.shape_groups[index]
                break
        non_zeros = group.cells.nonzero()
        for index in range(len(non_zeros[0])):
            self.problem[non_zeros[0][index] + group.top_left_x][non_zeros[1][index] + group.top_left_y] = 0
