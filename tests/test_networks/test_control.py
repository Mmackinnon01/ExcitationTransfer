import numpy as np
import pytest
from qgene.individual import Individual

from qtransfer.network import ControlSequence


@pytest.fixture
def test_sequence():
    return ControlSequence(times = [1], operators = [np.array([[1,0],[0,1]])])


@pytest.fixture
def test_individual():
    seq = ControlSequence(times = [1,2], operators = [np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]])])
    return seq.to_individual()


def test_setup():
    ControlSequence(times = [1], operators = [np.array([[1,0],[0,1]])])

def test_null_control():
    c = ControlSequence([], [])

def test_control_to_individual(test_sequence):
    i = test_sequence.to_individual()
    assert isinstance(i, Individual)
    assert i.n_genes == 3

def test_control_to_individual_with_mask(test_sequence):
    mask = (np.array([[1, 0],[0,0]]) == 1)
    i = test_sequence.to_individual(mask)
    assert i.n_genes == 1
    assert i.genome[0] == 1

def test_individual_to_control(test_individual):
    seq = ControlSequence.from_individual([1,2], test_individual, 2)
    assert pytest.approx(seq.operators[0]) == np.eye(2)
    assert pytest.approx(seq.operators[1]) == np.array([[0,1],[1,0]])
    assert len(seq.operators) == 2

def test_individual_to_control_with_mask(test_sequence):
    mask = (np.array([[1, 0],[0,0]]) == 1)
    i = test_sequence.to_individual(mask)
    seq = ControlSequence.from_individual([1], i, 2, control_mask = np.array([[True, False], [False, False]]))
    assert pytest.approx(seq.operators[0]) == np.array([[1,0],[0,0]])
    assert len(seq.operators) == 1