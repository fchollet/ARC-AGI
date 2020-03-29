import unittest
from ..models.SingleProblemSolver import SingleProblemSolver
from ..models.Groups.Group import Group


def get_group1c1():
    group1c1 = Group(2, 2, 1)
    group1c1.add_new_cell(1, 0, 1)
    group1c1.add_new_cell(0, 1, 1)
    group1c1.add_new_cell(3, 3, 1)
    return group1c1


def get_group2c1():
    group2c1 = Group(5, 5, 1)
    group2c1.add_new_cell(5, 4, 1)
    group2c1.add_new_cell(4, 5, 1)
    group2c1.add_new_cell(4, 4, 1)
    return group2c1


def get_group3c1():
    group3c1 = Group(2, 3, 1)
    group3c1.add_new_cell(2, 4, 1)
    group3c1.add_new_cell(2, 5, 1)
    group3c1.add_new_cell(2, 6, 1)
    return group3c1


def get_group1c2():
    # different colored group note the z axis
    group1c2 = Group(3, 2, 2)
    group1c2.add_new_cell(4, 2, 2)
    group1c2.add_new_cell(5, 2, 2)
    group1c2.add_new_cell(6, 2, 2)
    return group1c2


class TestFindGroupDifferences(unittest.TestCase):

    def test_no_groups_created_or_destroyed(self):
        groups = [get_group1c1(), get_group1c2(), get_group2c1(), get_group3c1()]
        groups2 = [get_group1c1(), get_group1c2(), get_group2c1(), get_group3c1()]
        sps = SingleProblemSolver({'train': [None], 'test': [None]})
        groups_changed = sps.find_group_differences([], [], groups, groups2)
        self.assertEqual(groups_changed['created'], [])
        self.assertEqual(groups_changed['destroyed'], [])
        self.assertEqual(len(groups_changed['not_modified']), len(groups))

    def test_one_group_created_none_destroyed(self):
        groups = [get_group1c2(), get_group2c1(), get_group3c1()]
        groups2 = [get_group1c1(), get_group1c2(), get_group2c1(), get_group3c1()]
        sps = SingleProblemSolver({'train': [None], 'test': [None]})
        groups_changed = sps.find_group_differences([], [], groups, groups2)
        self.assertEqual(groups_changed['destroyed'], [])
        self.assertTrue(groups_changed['created'][0].compare(get_group1c1()))
        self.assertEqual(len(groups_changed['not_modified']), len([get_group1c2(), get_group2c1(), get_group3c1()]))

    def test_no_groups_created_one_destroyed(self):
        groups = [get_group1c1(), get_group1c2(), get_group2c1(), get_group3c1()]
        groups2 = [get_group1c2(), get_group2c1(), get_group3c1()]
        sps = SingleProblemSolver({'train': [None], 'test': [None]})
        groups_changed = sps.find_group_differences([], [], groups, groups2)
        self.assertEqual(groups_changed['created'], [])
        self.assertTrue(groups_changed['destroyed'][0].compare(get_group1c1()))
        self.assertEqual(len(groups_changed['not_modified']), len([get_group1c2(), get_group2c1(), get_group3c1()]))

    def test_several_groups_created_and_destroyed(self):
        groups = [get_group1c1(), get_group1c2()]
        groups2 = [get_group2c1(), get_group3c1()]
        sps = SingleProblemSolver({'train': [None], 'test': [None]})
        groups_changed = sps.find_group_differences([], [], groups, groups2)
        self.assertEqual(len(groups_changed['created']), 2)
        self.assertTrue(len(groups_changed['destroyed']), 2)
        self.assertEqual(groups_changed['not_modified'], [])


class TestDetermineGroupChangeCause(unittest.TestCase):

    def test_cause_is_color(self):
        groups_changed = {
            'created': [],
            'destroyed': [get_group1c2()],
            'not_modified': [get_group1c1(), get_group2c1(), get_group3c1()]
        }
        sps = SingleProblemSolver({'train': [None], 'test': [None]})
        potential_cause = sps.determine_group_change_cause(groups_changed)

        self.assertEqual(potential_cause['created'], {})
        self.assertEqual(potential_cause['destroyed'], {})
        self.assertEqual(potential_cause['not_modified'], {'color': 1})

    def test_no_detected_common_cause(self):
        groups_changed = {
            'created': [get_group1c1(), get_group1c2(), get_group3c1()],
            'destroyed': [get_group1c2()],
            'not_modified': [get_group1c1(), get_group1c2(), get_group3c1()]
        }
        sps = SingleProblemSolver({'train': [None], 'test': [None]})
        potential_cause = sps.determine_group_change_cause(groups_changed)

        self.assertEqual(potential_cause['created'], {})
        self.assertEqual(potential_cause['destroyed'], {})
        self.assertEqual(potential_cause['not_modified'], {})


class TestCompareGroupChangeCauses(unittest.TestCase):
    """
    Oh god actually doing test driven development. I don't like it.

    Desired behavior:
    Get a list of possible traits like Color: value from each group.
    If the color: value key pair is the same in all cases then we assume that is correct and we are done

    If the color key is present in some cases but not in all cases, we throw it out.

    If the color key is present in all cases but the value changes then we look for commonalities between the groups

    Right now we are just assuming the number of shapes with the given key regardless of value may be of interest
    """
    def test_no_known_viable_causes(self):
        pass

    def test_color_key_and_value(self):
        pass

    def test_color_key_and_all_but_most_common_color_deleted(self):
        pass

    def test_color_key_and_most_common_color_deleted(self):
        pass

    def test_color_key_least_common_color_deleted(self):
        pass

    def test_color_key_all_but_least_common_color_deleted(self):
        pass


class TestCheckChangesOnInputs(unittest.TestCase):

    def test_single_solution_that_does_not_work(self):
        pass

    def test_single_solution_that_does_work(self):
        pass

    def test_4_solutions_where_3_work(self):
        pass


class TestApplyBestChangesToTest(unittest.TestCase):

    def test_apply_single_change_set_to_test(self):
        pass

    def test_apply_3_changesets_to_test(self):
        pass

    def test_apply_5_changesets_to_test(self):
        """
        Sorts the solutions that solve it with the fewest number of transforms
        :return:
        """
