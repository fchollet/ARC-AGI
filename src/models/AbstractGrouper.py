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


class AbstractGrouper:

    def __init__(self, problem):
        self.frames = self.generate_groups(problem)

    def generate_groups(self, problem):
        """
        :param problem: A problem retrieved by the problem fetcher
        :return: [frame]
        """
        return problem
