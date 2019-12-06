from src.models.problem_fetcher import ProblemFetcher
import pytest
import json


def test_fetch_specific_training_problem():
    """
    There is nothing to test yet this is just a place holder to make the github pipeline pass
    :return:
    """
    p_fetcher = ProblemFetcher()
    data = p_fetcher.get_specific_training_problem('0a938d79.json')
    assert '{"train": [{"input": [[' in json.dumps(data)


def test_fetch_specific_evaluation_problem():
    """
    There is nothing to test yet this is just a place holder to make the github pipeline pass
    :return:
    """
    p_fetcher = ProblemFetcher()
    data = p_fetcher.get_specific_evaluation_problem('0b17323b.json')
    assert '{"train": [{"input": [[' in json.dumps(data)


def test_fetch_random_training_problem():
    """
    There is nothing to test yet this is just a place holder to make the github pipeline pass
    :return:
    """
    p_fetcher = ProblemFetcher()
    data = p_fetcher.get_random_training()
    assert '{"train": [{"input": [[' in json.dumps(data)