from flask import Flask, render_template
from src.models.problem_fetcher import ProblemFetcher
app = Flask(__name__)


@app.route('/')
def hello():
    """
    Renders the default chollet template with no modifications
    :return: A rendered flask template
    """
    return render_template('testing_interface.html')


if __name__ == '__main__':
    """main method"""
    app.run()