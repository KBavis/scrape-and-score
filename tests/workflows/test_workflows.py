from scrape_and_score.workflows.workflows import predict, upcoming, historical, results
from unittest.mock import MagicMock
from unittest.mock import patch



@patch('scrape_and_score.workflows.workflows.nn_preprocess.preprocess')
def test_predict_workflows_invokes_functions():
    return None





