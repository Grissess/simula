import operator, functools
import numpy as np

from model import *
from lib import SimSchema
from sim import Property, Simulator

class NPZeroes(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPZRS', [
            Connector('S', list),
            Connector('T', str),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.zeros(node.getIn('S'), node.getIn('T')))

class NPZeroesLike(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPZRL', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.zeros_like(node.getIn('I')))

class NPOnes(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPONS', [
            Connector('S', list),
            Connector('T', str),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.ones(node.getIn('S'), node.getIn('T')))

class NPOnesLike(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPONL', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.ones_like(node.getIn('I')))

class NPAdd(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPADD', [
            Connector('I', np.ndarray, False),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', functools.reduce(operator.add, node.getIns('I')))

class NPMultiply(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPMUL', [
            Connector('I', np.ndarray, False),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', functools.reduce(operator.mul, node.getIns('I')))

class NPElementAdd(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPEADD', [
            Connector('A', np.ndarray),
            Connector('B', None, False),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') + functools.reduce(operator.add, node.getIns('B')))

class NPElementMultiply(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPEMUL', [
            Connector('A', np.ndarray),
            Connector('B', None, False),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') * functools.reduce(operator.mul, node.getIns('B')))

class NPElementMinimum(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPEMIN', [
            Connector('A', np.ndarray),
            Connector('B', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.minimum(node.getIn('A'), node.getIn('B')))

class NPElementMaximum(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPEMAX', [
            Connector('A', np.ndarray),
            Connector('B', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.maximum(node.getIn('A'), node.getIn('B')))

class NPElementSine(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPESIN', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.sin(node.getIn('I')))

class NPElementCosine(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPECOS', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.cos(node.getIn('I')))

class NPElementTangent(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPETAN', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.tan(node.getIn('I')))

class NPElementSinc(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPESINC', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.sinc(node.getIn('I')))

class NPLinSpace(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPLINS', [
            Connector('B', float),
            Connector('E', float),
            Connector('N', int),
            Connector('T', str),
            Connector('I', bool),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.linspace(node.getIn('B'), node.getIn('E'), node.getIn('N'), dtype=node.getIn('T'), endpoint=node.getIn('I')))

class NPARange(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPARNG', [
            Connector('B', float),
            Connector('E', float),
            Connector('S', float),
            Connector('T', str),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.arange(node.getIn('B'), node.getIn('E'), node.getIn('S'), dtype=node.getIn('T')))

class NPSum(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPSUM', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', float)
        ])

    def step(self, node):
        node.setOut('Q', np.sum(node.getIn('I')))

class NPBlackman(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPBKMN', [
            Connector('N', int),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.blackman(node.getIn('N')))

class NPConvolve(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPCVLV', [
            Connector('A', np.ndarray),
            Connector('B', np.ndarray),
            Connector('M', str),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.convolve(node.getIn('A'), node.getIn('B'), node.getIn('M')))

class NPContinuousConvolution(SimSchema):
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPCCVL', [
            Connector('A', np.ndarray),
            Connector('B', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def reset(self, node):
        node.overflow = np.array([], dtype=np.float64)
        node.prevsig = node.overflow

    def step(self, node):
        sig = node.getIn('A')
        win = node.getIn('B')
        if len(sig) and len(win):
            result = np.convolve(np.concatenate((node.prevsig, sig)), win)[len(node.prevsig):]
        else:
            result = sig
        convsz = min(len(sig), len(node.overflow))
        lerpa = np.linspace(0.0, 1.0, convsz)
        result[:convsz] = lerpa * result[:convsz] + (1 - lerpa) * node.overflow[:convsz]
        node.overflow = result[len(sig):]
        node.prevsig = sig[-convsz:]
        node.setOut('Q', result[:len(sig)])
