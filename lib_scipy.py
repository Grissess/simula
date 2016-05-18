import numpy as np
from scipy import signal

from model import *
from lib import SimSchema
from sim import Property, Simulator

class SPSquareWave(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'SciPy', 'SPSQW', [
            Connector('T', np.ndarray),
            Connector('D', float),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', signal.square(node.getIn('T'), node.getIn('D')))

class SPSawtoothWave(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'SciPy', 'SPSTW', [
            Connector('T', np.ndarray),
            Connector('W', float),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', signal.sawtooth(node.getIn('T'), node.getIn('W')))

class SPIIRFilter(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'SciPy', 'SPIIRF', [
            Connector('N', int),
            Connector('W'),
            Connector('P', float),
            Connector('S', float),
            Connector('B', str),
            Connector('F', str),
        ], [
            Connector('B', np.ndarray),
            Connector('A', np.ndarray),
        ])

    def step(self, node):
        b, a = signal.iirfilter(node.getIn('N'), node.getIn('W'), node.getIn('P'), node.getIn('S'), btype=node.getIn('B'), ftype=node.getIn('F'))
        node.setOut('B', b)
        node.setOut('A', a)

class SPFIRWin(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'SciPy', 'SPFIRF', [
            Connector('N', int),
            Connector('C', float),
            Connector('W', list),
        ], [
            Connector('B', np.ndarray),
        ])

    def step(self, node):
        node.setOut('B', signal.firwin(node.getIn('N'), node.getIn('C'), window=tuple(node.getIn('W'))))

class SPKaiserOrd(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'SciPy', 'SPKSOR', [
            Connector('R', float),
            Connector('W', float),
        ], [
            Connector('N', int),
            Connector('B', float),
        ])

    def step(self, node):
        N, beta = signal.kaiserord(node.getIn('R'), node.getIn('W'))
        node.setOut('N', N)
        node.setOut('B', beta)

class SPLFilter(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'SciPy', 'SPLFL', [
            Connector('X', np.ndarray),
            Connector('B', np.ndarray),
            Connector('A', None),
        ], [
            Connector('Y', np.ndarray),
        ])

    def reset(self, node):
        node.state = None

    def step(self, node):
        b, a, x = node.getIn('B'), node.getIn('A'), node.getIn('X')
        if node.state is None:
            node.state = signal.lfilter_zi(b, a)
        result = signal.lfilter(b, a, x, zi=node.state)
        if node.state is None:
            data = result
            node.state = signal.lfiltic(b, a, data, x)
        else:
            data, node.state = result
        print(x.shape, x.dtype, data.shape, data.dtype)
        node.setOut('Y', data)

class SPFiltFilt(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'SciPy', 'SPFTFT', [
            Connector('X', np.ndarray),
            Connector('B', None),
            Connector('A', np.ndarray),
        ], [
            Connector('Y', np.ndarray),
        ])

    def step(self, node):
        b, a, x = node.getIn('B'), node.getIn('A'), node.getIn('X')
        node.setOut('Y', signal.filtfilt(b, a, x))

class SPFFTConvolve(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'SciPy', 'SPFTCV', [
            Connector('A', np.ndarray),
            Connector('B', np.ndarray),
            Connector('M', str),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', signal.fftconvolve(node.getIn('A'), node.getIn('B'), node.getIn('M')))

