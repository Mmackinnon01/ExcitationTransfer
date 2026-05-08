import numpy as np
import pytest
from qsim.lin_alg import Operator, sigmaX
from qsim.state import DensityMatrix

from qtransfer.network import ControlSequence, Network


@pytest.fixture
def control():
    return ControlSequence([5], [sigmaX.matrix])

@pytest.fixture
def trial_network(control):
    return Network(
        A=np.array([[0, 1], [1, 0]]),
        rho0=DensityMatrix(np.array([[1, 0], [0, 0]])),
        control=control,
        sink_coupling_strengths = [1],
        local_loss_strengths = [1, 1],
    )

@pytest.fixture
def no_sink_trial_network(control):
    return Network(
        A=np.array([[0, 1], [1, 0]]),
        rho0=DensityMatrix(np.array([[1, 0], [0, 0]])),
        control = control,
        sink_coupling_strengths = [],
        local_loss_strengths = [1, 1],
    )


def test_network_creation(trial_network):
    assert pytest.approx(trial_network.A) == np.array([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])
    assert len(trial_network.jumps) == 2
    assert trial_network.n_sinks == 1
    print(trial_network.H_t(0).matrix)
    assert trial_network.H_t(0) == Operator(np.array([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]))


def test_evolve(no_sink_trial_network):
    assert len(no_sink_trial_network.state_data.values()) == 2
    assert -1 in no_sink_trial_network.state_data.keys()
    for state in no_sink_trial_network.state_data.values():
        assert state.isLegitimate()

