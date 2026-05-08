import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qsim.dynamics import Dynamics, GKSLGenerator, IVPPropagator, LiouvillianGenerator
from qsim.lin_alg import Operator, TOperator
from qsim.logging import Logger
from qsim.state import DensityMatrix

from .animate import animateConnections, animateConnectionsSquare, animateExcitationFlow
from .control import ControlSequence


class Network:

    def __init__(
        self,
        A: np.ndarray,
        rho0: DensityMatrix,
        control: ControlSequence = ControlSequence([], []),
        sink_locations: list[int] = [],
        sink_coupling_strengths: list[float] = [],
        local_loss_strengths: list[float] = []
    ) -> None:
        self._state_data = None

        self.n_network = A.shape[0]
        self.n_sinks = len(sink_coupling_strengths)

        self.sink_locations = sink_locations
        self.sink_coupling_strengths = sink_coupling_strengths

        self.A = self.addSinksAndEnv(A)
        self.A_with_sinks = self.addSinksAndEnv(A)
        for i, n in enumerate(sink_locations):
            self.A_with_sinks[self.n_network + i, n] = 1
            self.A_with_sinks[n, self.n_network + i] = 1

        self.local_loss_strengths = local_loss_strengths
        self.rho0 = DensityMatrix(self.addSinksAndEnv(rho0.matrix))

        self.control = control

        self.H_t = Operator(self.A) + controlHamiltonian(control.times, [self.addSinksAndEnv(op) for op in control.operators])
        self.setupDynamics()

    def setupDynamics(self):
        self.setupJumps()
        self.prop = IVPPropagator()
        self.gen = GKSLGenerator(H = self.H_t , jumps=self.jumps)
        self.gen_l = LiouvillianGenerator.fromGKSL(self.gen)
        self.log = Logger(log_state=True)
        self.dynam = Dynamics(self.prop, self.gen)
        self.dynam.addCallback(self.log.log)

    def setupJumps(self):
        #Environmental damping
        self._jumps = []

        if len(self.local_loss_strengths) > 0:
            for i in range(self.n_network):
                jump = np.zeros(shape=(self.n_network+1+self.n_sinks,)*2).astype(np.complex128)
                jump[self.n_network + self.n_sinks, i] = self.local_loss_strengths[i]
                self._jumps.append(Operator(jump))

        #Sink damping
        for i, n in enumerate(self.sink_locations):
            jump = np.zeros(shape=(self.n_network+1+self.n_sinks,)*2).astype(np.complex128)
            jump[self.n_network + i, n] = self.sink_coupling_strengths[i]
            self._jumps.append(Operator(jump))

    def addSinksAndEnv(self, matrix):
        extended_mat = np.zeros(shape=(matrix.shape[0] + 1 + self.n_sinks,)*2).astype(np.complex128)
        extended_mat[:matrix.shape[0], :matrix.shape[1]] = matrix
        return extended_mat

    @property
    def jumps(self) -> list[Operator]:
        return self._jumps

    def evolve(self, log_times: list[float]) -> None:
        self.log.clear()
        self.dynam.evolve(self.rho0, ts=log_times)

    def steadyState(self, t: float, state: DensityMatrix) -> DensityMatrix:
        return self.gen_l.steadyState(t, state)

    @property
    def state_data(self):
        if self._state_data is None:
            
            if len(self.control.times)>0:
                self.evolve(self.control.times)
                self._state_data = self.log.state_log
                self._state_data[-1] = self.steadyState(self.control.times[-1], self._state_data[self.control.times[-1]])
            else:
                self._state_data = {-1:self.steadyState(0, self.rho0)}
        return self._state_data

    def animateEvolutionWithConnections(self, ts, n_steps):
        times = [ts * step for step in range(n_steps)]
        self.evolve(times)
        states = [state.matrix for state in self.log.state_log.values()]
        adjs = [self.H_t(t).matrix for t in times]
        return animateConnections(adjs, states, ts, self.sink_locations)

    def animateEvolutionWithConnectionsSquare(self, ts, n_steps):
        times = [ts * step for step in range(n_steps)]
        self.evolve(times)
        states = [state.matrix for state in self.log.state_log.values()]
        adjs = [self.H_t(t).matrix for t in times]
        return animateConnectionsSquare(adjs, states, ts, self.sink_locations)

    def animateEvolutionWithExcitationFlow(self, ts, n_steps):
        times = [ts * step for step in range(n_steps)]
        self.evolve(times)
        states = [state.matrix for state in self.log.state_log.values()]
        adjs = [self.H_t(t).matrix for t in times]
        return animateExcitationFlow(adjs, states, ts, self.sink_locations)

    @property
    def energy_transfer(self):
        energy_captured = np.sum(np.diag(self.state_data[-1].matrix)[self.n_network : self.n_network + self.n_sinks])

        return energy_captured .real

    @property
    def control_cost(self):
        costs = [0]

        for t in self.control.times:
            state = self.state_data[t]
            delta_h = self.H_t(t) - self.H_t(t - 0.01)
            costs.append((state @ delta_h).trace())

        return sum(costs).real

    def diagram(self):
        fig, ax = plt.subplots()
        G = nx.from_numpy_array(self.A_with_sinks)
        pos = nx.circular_layout(G)

        nx.draw(G, pos, with_labels=True, node_color="lightgreen", ax = ax)
        return fig

def controlHamiltonian(control_times, adjustments):
    components = []

    def make_control_func(t_start, t_end=None):
        if t_end is None:
            return lambda t: 1 if t >= t_start else 0
        else:
            return lambda t: 1 if (t_start <= t < t_end) else 0

    for i, (t_c, adj) in enumerate(zip(control_times, adjustments)):
        if i == len(control_times) - 1:
            components.append(make_control_func(t_c) * TOperator.from_static(Operator(adj)))
        else:
            components.append(make_control_func(t_c, control_times[i+1]) * TOperator.from_static(Operator(adj)))
        
    
    return sum(components)

