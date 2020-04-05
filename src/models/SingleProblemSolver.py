"""
Responsible for taking in a problem and attempting to solve it
"""
from .Groups.Grouper import Grouper
import numpy as np


class SingleProblemSolver:

    def __init__(self, problem):
        """

        :param problem: A dict of format {'train': [{'input': [][], 'output':[][]}],
                                          'test': [{'input': [][], 'output': [][]}]
                                          }
                        We can assume that train has several elements in its list and that test only has one.
        """
        self.training = problem['train']
        self.test = problem['test'][0]

    def solve(self):
        groups_changed_list = []
        group_change_causes = []
        board_differences = []
        for train in self.training:
            in_g = Grouper(train['input'])
            out_g = Grouper(train['output'])
            groups_changed = self.find_group_differences(
                in_g.shape_groups, out_g.shape_groups, in_g.color_groups, out_g.color_groups)
            board_differences.append(self.find_board_differences(train['input'], train['output'], in_g, out_g))
            groups_changed_list.append(groups_changed)
            group_change_causes.append(self.determine_group_change_cause(groups_changed, in_g.metadata))
        potential_change_rules = self.compare_group_change_causes(group_change_causes)
        final_board_differences = self.compare_board_differences(board_differences)
        final_change_rules = self.check_changes_on_input(potential_change_rules)
        output = self.apply_best_changes_to_test(final_change_rules, final_board_differences)
        return output

    def find_board_differences(self, board_in, board_out, in_grouper, out_grouper):
        """
        Find differences in the overall board state. Was the whole board rotated? Shifted? Did it shrink to the size
        of the groups left?
        :param board_in:
        :param board_out:
        :param in_grouper:
        :param out_grouper:
        :return:
        """
        board_differences = []
        np_in = np.array(board_in, np.int32)
        np_out = np.array(board_out)
        if np_in.shape[0] > np_out.shape[0] and np_in.shape[1] > np_out.shape[1]:
            # Shrink it to size of the object. Don't get fancy. later we will have to
            board_differences.append('shrink_to_object')
        return board_differences

    def compare_board_differences(self, board_differences):
        final_board_differences = set(board_differences[0])
        for i in range(1, len(board_differences)):
            final_board_differences = final_board_differences.intersection(board_differences[i])
            if not final_board_differences:
                break
        return final_board_differences


    def find_group_differences(self, in_shape_groups, out_shape_groups, in_color_groups, out_color_groups):
        """
        FOR RIGHT NOW JUST WORRYING ABOUT COLORED SHAPES

        :param in_shape_groups:
        :param in_color_groups:
        :param out_shape_groups:
        :param out_color_groups:
        :return: groups_changed TODO eventually this will become all of the deltas we would expect to
         see like shapes moved, changed color, reflected etc.
         TODO the double for loop and other pieces of this are not efficent. We need to make this more efficent later
        """
        groups_created = []
        groups_destroyed = []
        groups_not_modified = []
        for in_color_group in in_color_groups:
            destroyed = False
            for out_color_group in out_color_groups:
                if in_color_group.compare(out_color_group):
                    groups_not_modified.append(in_color_group)
                    destroyed = True
                    break
            if not destroyed:
                groups_destroyed.append(in_color_group)

        for out_color_group in out_color_groups:
            created = False
            for in_color_group in in_color_groups:
                if in_color_group.compare(out_color_group):
                    created = True
                    break
            if not created:
                groups_created.append(out_color_group)
        groups_changed = {
            'created': groups_created,
            'destroyed': groups_destroyed,
            'not_modified': groups_not_modified
        }
        return groups_changed

    def determine_group_change_cause(self, groups_changed, in_metadata):
        """
        :param groups_changed: {'Change_type':[group]}
        :return: causes dict {'change_type: {potential_cause: value}}
        I.E.
        {'created': {color: 2, num_groups: 5}, 'destroyed': {num_groups}, 'not_modified': {color: 1, num_groups: 1}}
        """
        potential_cause = {
            'created': {},
            'destroyed': {},
            'not_modified': {}
        }
        invalidated_keys = set()
        for change, groups in groups_changed.items():
            if len(groups) > 0:
                # change is something like not_modified
                if 'color' not in invalidated_keys:
                    potential_cause, invalidated_keys = self.determine_group_change_cause_color_helper(
                        potential_cause, change, groups, invalidated_keys, in_metadata)
        return potential_cause

    def determine_group_change_cause_color_helper(self, potential_cause, change, groups, invalidated_keys, in_metadata):
        if 'color' not in potential_cause[change]:
            potential_cause[change]['color'] = [groups[0].color]
        for group in groups:
            if potential_cause[change]['color'] != group.color:
                invalidated_keys.add('color')
                del potential_cause[change]['color']
                return potential_cause, invalidated_keys
        if groups[0].color == in_metadata['most_common_color']:
            potential_cause[change]['color'].append('most_common_color')
        elif groups[0].color == in_metadata['least_common_color']:
            potential_cause[change]['color'].append('least_common_color')
        return potential_cause, invalidated_keys

    def compare_group_change_causes(self, group_change_causes):
        """

        :param group_change_causes:
        [{
            'not_modified': {
                'color': [value, 'least_common']
            }
        }]
        :return: A dictionary of changes
        {delete: 'color': value} or {Not_modified: 'color': {Largest}} or {'move': {'Type': 'line'}}
        {'delete': {'color': 'most_common'}}
        """
        potential_causes = group_change_causes[0]
        for index in range(1, len(group_change_causes)):
            # for each training group. Check that all of the potential causes are in both places.
            # if the key is missing from either place than it can be removed.
            # if the key and value both match than we leave it
            # if the key matches but the value doesn't then we need to call a new helper function while all of the key
            # value pairs don't match.
            # that helper will return commonalities between
            for group_mod_key, modification_dict in group_change_causes[index].items():
                # group_mod_key would be something like created, destroyed or not_modified
                potential_causes[group_mod_key], modification_dict = self.remove_entries_from_dicts_with_no_shared_key(
                    potential_causes[group_mod_key], modification_dict)
                for mod_action_key, modification_list in modification_dict.items():
                    # mod_action_key is going to be something like color. We know it is in potential_causes
                    potential_causes[group_mod_key][mod_action_key] = list(set(
                        potential_causes[group_mod_key][mod_action_key]).intersection(modification_list))
                    if not potential_causes[group_mod_key][mod_action_key]:
                        # if the list came back empty. Delete it.
                        del potential_causes[group_mod_key][mod_action_key]
        return potential_causes

    def remove_entries_from_dicts_with_no_shared_key(self, potential_causes, modification_dict):
        """
        Takes in two dictionaries and removes all of the entries in both that don't have the same key in the other dict
        :return:
        """
        # delete all of the values that don't have a shared key
        potential_keys = potential_causes.keys()
        current_keys = modification_dict.keys()
        potential_causes = {your_key: potential_causes[your_key] for your_key in current_keys if your_key in potential_causes}
        modification_dict = {your_key: modification_dict[your_key] for your_key in potential_keys if your_key in modification_dict}
        return potential_causes, modification_dict

    def check_changes_on_input(self, potential_changes):
        """
        Try to use the change set on the input.
        If it doesn't work move it to a different data structure to only be used if we don't have anything that works
        :param potential_changes:
        :return:
        """
        return potential_changes

    def apply_best_changes_to_test(self, potential_changes, final_board_differences):
        """
        :param potential_changes:
        {
            'destroyed': {
                'color': [value, 'least_common']
            }
        }
        :return:
        """
        input_groups = Grouper(self.test['input'])

        if 'destroyed' in potential_changes:
            groups_to_be_removed = []
            if 'color' in potential_changes['destroyed']:
                for value in potential_changes['destroyed']['color']:
                    if isinstance(value, int):
                        groups_to_be_removed += self.get_group_with_color(input_groups.color_groups, value)
                    elif value == 'most_common_color':
                        groups_to_be_removed += self.get_group_with_color(
                            input_groups.color_groups, input_groups.metadata['most_common_color'])
                    elif value == 'least_common_color':
                        groups_to_be_removed += self.get_group_with_color(
                            input_groups.color_groups, input_groups.metadata['most_common_color'])
            for group in groups_to_be_removed:
                input_groups.remove_group(group)

        if 'shrink_to_object' in final_board_differences and len(
                input_groups.shape_groups + input_groups.color_groups) > 0:
            # go through each object find the values closest to the walls using bounding boxes and top left xy values
            # you could do this with numpy but this should be less algorithmic work
            new_top_left_x = 9999
            new_top_left_y = 9999
            bottom_right_x = 0
            bottom_right_y = 0
            for group in input_groups.shape_groups + input_groups.color_groups:
                if new_top_left_x > group.top_left_x:
                    new_top_left_x = group.top_left_x
                if new_top_left_y > group.top_left_y:
                    new_top_left_y = group.top_left_y
                btm_rt_x = group.top_left_x + group.bounding_box_x_len
                btm_rt_y = group.top_left_y + group.bounding_box_y_len
                if bottom_right_x < btm_rt_x:
                    bottom_right_x = btm_rt_x
                if bottom_right_y < btm_rt_y:
                    bottom_right_y = btm_rt_y
            for group in input_groups.shape_groups + input_groups.color_groups:
                group.top_left_x -= new_top_left_x
                group.top_left_y -= new_top_left_y
            output_grid = np.zeros((bottom_right_x + 1 - new_top_left_x, bottom_right_y + 1 - new_top_left_y),
                                   dtype=np.int32)
        else:
            output_grid = np.zeros(input_groups.problem.shape, dtype=np.int32)

        for group in input_groups.shape_groups + input_groups.color_groups:
            output_grid[group.top_left_x: group.top_left_x + group.bounding_box_x_len + 1,
                        group.top_left_y: group.bounding_box_y_len + group.top_left_y + 1] = group.cells

        return output_grid

    def get_group_with_color(self, input_groups, color):
        groups_w_color = []
        for group in input_groups:
            if group.color == color:
                groups_w_color.append(group)
        return groups_w_color

