import numpy as np
import pytest
from trustpy.cond_nts import CNTS

@pytest.fixture
def oracle():
    return np.array([0, 1, 1, 0])

@pytest.fixture
def softmax_preds():
    return np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.9, 0.1],
    ])

def test_valid_computation(oracle, softmax_preds):
    cnts = CNTS(oracle, softmax_preds, show_summary=False, export_summary=False)
    result = cnts.compute()
    assert isinstance(result, dict)
    assert "overall" in result
    assert all(k.startswith("class_") or k == "overall" for k in result)

def test_valid_input(oracle, softmax_preds):
    cnts = CNTS(oracle, softmax_preds)
    assert isinstance(cnts, CNTS)

def test_oracle_not_ndarray(softmax_preds):
    with pytest.raises(AssertionError, match="Oracle/Actual Classes must be a NumPy array"):
        CNTS([0, 1, 1], softmax_preds)

def test_predictions_not_ndarray(oracle):
    with pytest.raises(AssertionError, match="Predictions/Predicted Classes must be a NumPy array"):
        CNTS(oracle, [[0.9, 0.1], [0.2, 0.8]])

def test_alpha_not_numeric(oracle, softmax_preds):
    with pytest.raises(AssertionError, match="alpha must be a number"):
        CNTS(oracle, softmax_preds, alpha="high")

def test_beta_not_numeric(oracle, softmax_preds):
    with pytest.raises(AssertionError, match="beta must be a number"):
        CNTS(oracle, softmax_preds, beta=None)

def test_trust_spectrum_not_bool(oracle, softmax_preds):
    with pytest.raises(AssertionError, match="trust_spectrum must be True/False"):
        CNTS(oracle, softmax_preds, trust_spectrum="yes")

def test_show_summary_not_bool(oracle, softmax_preds):
    with pytest.raises(AssertionError, match="show_summary must be True/False"):
        CNTS(oracle, softmax_preds, show_summary=1)

def test_export_summary_not_bool(oracle, softmax_preds):
    with pytest.raises(AssertionError, match="export_summary must be True/False"):
        CNTS(oracle, softmax_preds, export_summary="true")

def test_oracle_not_1d(softmax_preds):
    with pytest.raises(AssertionError, match="Oracle/Actual Classes must be a 1D array"):
        CNTS(np.array([[0], [1], [1]]), softmax_preds)

def test_predictions_not_2d(oracle):
    with pytest.raises(AssertionError, match="Predictions/Predicted Classes must be a 1D array"):
        CNTS(oracle, np.array([0.9, 0.1, 0.2]))

def test_sample_size_mismatch(softmax_preds):
    with pytest.raises(AssertionError, match="Number of samples mismatch"):
        CNTS(np.array([0, 1]), softmax_preds)

def test_predictions_out_of_bounds(oracle):
    bad_predictions = np.array([[1.2, -0.2], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    with pytest.raises(AssertionError, match="Predictions must be between 0 and 1"):
        CNTS(oracle, bad_predictions)

def test_predictions_not_softmax(oracle):
    bad_predictions = np.array([[0.5, 0.3], [0.2, 0.8], [0.3, 0.7], [0.4, 0.4]])
    with pytest.raises(AssertionError, match="Each row of SoftMax predictions must sum to 1"):
        CNTS(oracle, bad_predictions)

def test_known_output():
    oracle = np.array([0, 0, 1, 1])
    preds = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.4, 0.6],
    ])
    cnts = CNTS(oracle, preds, show_summary=False, export_summary=False)
    scores = cnts.compute()
    assert abs(float(scores["overall"]) - 0.775) < 1e-3

def test_all_incorrect():
    oracle = np.array([0, 0, 1, 1])
    preds = np.array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.7, 0.3],
    ])
    cnts = CNTS(oracle, preds, show_summary=False, export_summary=False)
    scores = cnts.compute()
    assert abs(float(scores["overall"]) - 0.2) < 1e-3

def test_single_sample():
    oracle = np.array([1])
    preds = np.array([[0.2, 0.8]])
    cnts = CNTS(oracle, preds, show_summary=False, export_summary=False)
    scores = cnts.compute()
    assert "overall" in scores
    assert isinstance(scores["overall"], str)
