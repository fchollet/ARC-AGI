from src.models import problem_fetcher
import pytest
import json


def test_fetch_specific_training_problem():
    """
    There is nothing to test yet this is just a place holder to make the github pipeline pass
    :return:
    """
    p_fetcher = problem_fetcher.ProblemFetcher()
    data = p_fetcher.get_specific_training_problem('0a938d79.json')
    assert '{"train": [{"input": [[' in json.dumps(data)

