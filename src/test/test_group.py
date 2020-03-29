"""
Tests the group class
"""

import unittest
from ..models.Groups.Group import Group


class TestGroup(unittest.TestCase):
    """
    Tests the group class
    """
    def test_add_new_cell_00_origin(self):
        """
        There is nothing to test yet this is just a place holder to make the github pipeline pass
        :return:
        """
        group = Group(0, 0, 1)

        self.assertEqual(group.bounding_box_y_len, 0)
        self.assertEqual(group.bounding_box_x_len, 0)
        self.assertEqual(group.top_left_x, 0)
        self.assertEqual(group.top_left_y, 0)

        group.add_new_cell(1, 0, 1)

        self.assertEqual(group.bounding_box_y_len, 0)
        self.assertEqual(group.bounding_box_x_len, 1)
        self.assertEqual(group.top_left_x, 0)
        self.assertEqual(group.top_left_y, 0)

        group.add_new_cell(0, 1, 1)
        self.assertEqual(group.bounding_box_y_len, 1)
        self.assertEqual(group.bounding_box_x_len, 1)
        self.assertEqual(group.top_left_x, 0)
        self.assertEqual(group.top_left_y, 0)

        group.add_new_cell(1, 1, 1)
        self.assertEqual(group.bounding_box_y_len, 1)
        self.assertEqual(group.bounding_box_x_len, 1)
        self.assertEqual(group.top_left_x, 0)
        self.assertEqual(group.top_left_y, 0)

    def test_add_new_cell_22_origin(self):
        group = Group(2, 2, 1)

        self.assertEqual(group.bounding_box_x_len, 0)
        self.assertEqual(group.bounding_box_y_len, 0)
        self.assertEqual(group.top_left_x, 2)
        self.assertEqual(group.top_left_y, 2)

        group.add_new_cell(1, 0, 1)

        self.assertEqual(group.bounding_box_x_len, 1)
        self.assertEqual(group.bounding_box_y_len, 2)
        self.assertEqual(group.top_left_x, 1)
        self.assertEqual(group.top_left_y, 0)

        group.add_new_cell(0, 1, 1)
        self.assertEqual(group.bounding_box_y_len, 2)
        self.assertEqual(group.bounding_box_x_len, 2)
        self.assertEqual(group.top_left_x, 0)
        self.assertEqual(group.top_left_y, 0)

        group.add_new_cell(6, 6, 1)
        self.assertEqual(group.bounding_box_y_len, 6)
        self.assertEqual(group.bounding_box_x_len, 6)
        self.assertEqual(group.top_left_x, 0)
        self.assertEqual(group.top_left_y, 0)
