"""
This is the base class for a group.
"""
import numpy as np


class Group:

    def __init__(self, x, y, color):
        """
        Takes in the location and color of the first cell in a group

        :param x: int The x position of the first cell
        :param y: int The y position of the first cell
        :param color: int The color of the first cell
        """
        if x < 0:
            raise ValueError('X must be a positive int')
        if y < 0:
            raise ValueError('Y must be a positive int')
        self.top_left_x = x
        self.top_left_y = y
        self.bounding_box_x_len = 0
        self.bounding_box_y_len = 0
        self.num_colored_cells = 1
        self.color = color

        # The cells are added via relative position to the top left corner.
        # This way later the groups can be compared without their relative positioning mattering.
        # TODO Must find a way to represent this in such a way that the matrix can be rotated 90 degrees or mirrored
        # over an arbitrary value and for a useful comparison to be made
        self.cells = np.zeros((1, 1))
        self.cells[0][0] = color

    def add_new_cell(self, x, y, color):
        """
        Adds a new cell to the group and updates all of the corresponding self params
        :param x: new x value of cell
        :param y: new y val of cell
        :param color: Color of new cell
        :return: Updates self
        """
        # if the origin changes then we are going to need to update all of the cells in the grid with new relative
        # positions.
        self.num_colored_cells += 1
        if color != self.color:
            self.color = -1
        x_origin_change = 0
        y_origin_change = 0
        bounding_box_change = False
        if x < self.top_left_x:
            x_origin_change = self.top_left_x - x
            self.top_left_x = x
            self.bounding_box_x_len += x_origin_change
            bounding_box_change = True
        elif x > self.top_left_x + self.bounding_box_x_len:
            self.bounding_box_x_len = x - self.top_left_x
            bounding_box_change = True
        if y < self.top_left_y:
            y_origin_change = self.top_left_y - y
            self.top_left_y = y
            self.bounding_box_y_len += y_origin_change
            bounding_box_change = True
        elif y > self.top_left_y + self.bounding_box_y_len:
            self.bounding_box_y_len = y - self.top_left_y
            bounding_box_change = True

        if bounding_box_change:
            new_cells = np.zeros((self.bounding_box_x_len + 1, self.bounding_box_y_len + 1))
            new_cells[x_origin_change:len(self.cells) + x_origin_change,
                      y_origin_change:len(self.cells[0]) + y_origin_change] = self.cells
            self.cells = new_cells
        self.cells[x - self.top_left_x][y - self.top_left_y] = color

    def compare(self, other_group):
        """
        Takes in another group and will eventually return a list of transforms to get from this to other_group if
        possible and true or false if they are equivilant or it is possible to transform this into that other object
        in less than a prespecified number of transforms
        :param other_group:
        :return: bool (EVENTUALLY TRANSFORMS)
        """
        x_bounds = self.bounding_box_x_len == other_group.bounding_box_x_len
        y_bounds = self.bounding_box_y_len == other_group.bounding_box_y_len
        same_num_cells = self.num_colored_cells == other_group.num_colored_cells
        if not (x_bounds and y_bounds and same_num_cells):
            return False
        for row_ind in range(len(other_group.cells)):
            for col_ind in range(len(other_group.cells[0])):
                if other_group.cells[row_ind][col_ind] != self.cells[row_ind][col_ind]:
                    return False
        return True

    def equals(self, other_group):
        if not self.top_left_x == other_group.top_left_x and self.top_left_y == other_group.top_left_y:
            return False
        return self.compare(other_group)
