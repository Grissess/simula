import operator, functools
import numpy as np

from model import *
from lib import SimSchema
from sim import Property, Simulator

class NPToArray(SimSchema):
    '''Construct an ndarray out of the object I, and emits it at Q.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NP2AR', [
            Connector('I'),
        ], [
            Connector('Q', np.ndarray)
        ])

    def step(self, node):
        node.setOut('Q', np.array(node.getIn('I')))

class NPZeroes(SimSchema):
    '''Constructs a zero-filled ndarray with shape S and data type T.'''
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
    '''Constructs a zero-filled ndarray with the same shape and data type as I.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPZRL', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.zeros_like(node.getIn('I')))

class NPOnes(SimSchema):
    '''Constructs a one-filled ndarray with shape S and data type T.'''
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
    '''Constructs a one-filled ndarray with the same shape and data type as I.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPONL', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.ones_like(node.getIn('I')))

class NPAdd(SimSchema):
    '''Adds ndarrays. The result is generally an element-wise addition, with broadcasting to resolve unused dimensions. (See the numpy documentation for details.)'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPADD', [
            Connector('I', np.ndarray, False),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', functools.reduce(operator.add, node.getIns('I')))

class NPMultiply(SimSchema):
    '''Multiplies ndarrays. The result is generally an element-wise multiplication, with broadcasting to resolve unused dimensions. (See the numpy documentation for details.)'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPMUL', [
            Connector('I', np.ndarray, False),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', functools.reduce(operator.mul, node.getIns('I')))

class NPElementAdd(SimSchema):
    '''Adds the sum of ndarrays with a constant, which will be broadcast over all dimensions. (See NPAdd.)'''
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
    '''Multiplies the product of ndarrays with a constant, which will be broadcast over all dimensions. (See NPMultiply.)'''
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
    '''Returns an ndarray where each element is the minimum of the input arrays, with smaller shapes broadcast over larger shapes. (See numpy documentation for details.)'''
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
    '''Returns an ndarray where each element is the maximum of the input arrays, with smaller shapes broadcast over larger shapes. (See numpy documentation for details.)'''
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
    '''Computes an ndarray for which Q[x] = sin(I[x]) for all possible vectors x.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPESIN', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.sin(node.getIn('I')))

class NPElementCosine(SimSchema):
    '''Computes an ndarray for which Q[x] = cos(I[x]) for all possible vectors x.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPECOS', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.cos(node.getIn('I')))

class NPElementTangent(SimSchema):
    '''Computes an ndarray for which Q[x] = tan(I[x]) for all possible vectors x.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPETAN', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.tan(node.getIn('I')))

class NPElementSinc(SimSchema):
    '''Computes an ndarray for which Q[x] = sinc(I[x]) for all possible vectors x.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPESINC', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.sinc(node.getIn('I')))

class NPLinSpace(SimSchema):
    '''Computes a linear space--a unidimensional, real ndarray beginning with B inclusive and ending at E (inclusive if I, otherwise exclusive) with data type T; the ndarray will always have shape (N,) (or, rather, N elements).'''
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
    '''Computes a range--a unidimensional, real ndarray beginning with B inclusive and ending at E exclusive, with the delta between subsequent elements equal to (or as approximately as possible equal to) S, with data type T. The number of elements is ideally floor((E - B) / S), but the precision of this result is limited by the implementation and data type. See NPLinSpace for an alternative with fixed dimension.'''
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
    '''Computes the sum of all elements of an ndarray.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPSUM', [
            Connector('I', np.ndarray),
        ], [
            Connector('Q', float)
        ])

    def step(self, node):
        node.setOut('Q', np.sum(node.getIn('I')))

class NPBlackman(SimSchema):
    '''Computes an N-point Blackman window.'''
    def __init__(self):
        SimSchema.__init__(self, 'NumPy', 'NPBKMN', [
            Connector('N', int),
        ], [
            Connector('Q', np.ndarray),
        ])

    def step(self, node):
        node.setOut('Q', np.blackman(node.getIn('N')))

class NPConvolve(SimSchema):
    '''Computes the convolution of A and B, with mode M, which may be 'full' for a complete, boundary-to-boundary convolution of size |A|+|B|-1, 'same' for a half-internal convolution of size max(|A|, |B|), or 'valid' for internal convlution of size max(|A|, |B|) - min(|A|, |B|) + 1.'''
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
    '''Does a continuous convolution over time of A and B: both are convolved fully, and the boundary effects from the latter end of one convolution are contributed to the boundary of the prior convolution on the next tick, giving a sliding window of convolution. For best results, both arrays should have constant size over the simulation.'''
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
