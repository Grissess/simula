import operator, functools, math, time
import xml.etree.ElementTree as ET

from model import *
from sim import Property, Simulator
from procon import ProducerRegistry, ConsumerRegistry

class SchemaRegistry(type):
    ALL = {}
    def __new__(mcs, name, bases, dict):
        tp = type.__new__(mcs, name, bases, dict)
        if name != 'SimSchema':
            inst = tp()
            if inst.name in mcs.ALL:
                print('WARNING: Multiple registrations for name', inst.name)
                print('  old registrant:', mcs.ALL[inst.name])
                print('  new registrant:', inst)
            mcs.ALL[inst.name] = inst
        return tp

class SimSchema(Schema, metaclass=SchemaRegistry):
    def __init__(self, cat, name, ins, outs, props=None):
        Schema.__init__(self, name, ins, outs)
        self.cat = cat
        self.props = props or []
        self.propmap = {prop.name: prop for prop in self.props}

    def reset(self, node):
        pass

    def step(self, node):
        raise NotImplementedError('SimSchema.step()')

################################################################################

class Add(SimSchema):
    '''Computes the sum of all of its inputs.'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'ADD', [
            Connector('I', float, False),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', sum(node.getIns('I')))

class Subtract(SimSchema):
    '''Computes Q = A - B'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'SUB', [
            Connector('A', float),
            Connector('B', float),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') - node.getIn('B'))

class Multiply(SimSchema):
    '''Computes the product of all of its inputs with floating point precision.'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'MUL', [
            Connector('I', float, False),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', functools.reduce(operator.mul, node.getIns('I'), 1.0))

class Divide(SimSchema):
    '''Computes Q = A / B with floating-point precision.'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'DIV', [
            Connector('A', float),
            Connector('B', float),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') / node.getIn('B'))

class DivideModulus(SimSchema):
    '''Computes Q (quotient) and M (modulus) such that Q*B + M = A and M is in [0, B) or (B, 0], depending on the sign of B.'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'DIMO', [
            Connector('A', float),
            Connector('B', float),
        ], [
            Connector('Q', float),
            Connector('M', float),
        ])

    def step(self, node):
        quot, mod = divmod(node.getIn('A'), node.getIn('B'))
        node.setOut('Q', quot)
        node.setOut('M', mod)

class Exponentiate(SimSchema):
    '''Computes Q = A ** B (A raised to the B).'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'EXP', [
            Connector('A', float),
            Connector('B', float),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') ** node.getIn('B'))

class Negate(SimSchema):
    '''Computes Q = -I.'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'NEG', [
            Connector('I', float),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', -node.getIn('I'))

class Reciprocal(SimSchema):
    '''Computes Q = 1 / I = I ** (-1).'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'RCP', [
            Connector('I', float),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', 1.0 / node.getIn('I'))

class Signum(SimSchema):
    '''Computes Q = signum(I), where signum is a function that returns a value in the set {-1, 0, 1} whose sign is the same as the input, or zero if the input is zero.'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'SGN', [
            Connector('I', float),
        ], [
            Connector('Q', int),
        ])

    def step(self, node):
        val = node.getIn('I')
        if val == 0.0:
            node.setOut('Q', 0.0)
        else:
            node.setOut('Q', math.copysign(1.0, val))

class Ceiling(SimSchema):
    '''Computes Q = ceil(I), where ceil is a function returning the least integer greater than I.'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'CEIL', [
            Connector('I', float),
        ], [
            Connector('Q', int),
        ])

    def step(self, node):
        node.setOut('Q', math.ceil(node.getIn('I')))

class Floor(SimSchema):
    '''Computes Q = floor(I), where floor is a function returning the greatest integer less than I.'''
    def __init__(self):
        SimSchema.__init__(self, 'Arithmetic', 'FLOR', [
            Connector('I', float),
        ], [
            Connector('Q', int),
        ])

    def step(self, node):
        node.setOut('Q', math.floor(node.getIn('I')))

################################################################################

class Zero(SimSchema):
    '''Emits a constant 0.0.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'ZERO', [], [Connector('0', float)])

    def step(self, node):
        node.setOut('0', 0.0)

class One(SimSchema):
    '''Emits a constant 1.0.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'ONE', [], [Connector('1', float)])

    def step(self, node):
        node.setOut('1', 1.0)

class Two(SimSchema):
    '''Emits a constant 2.0.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'TWO', [], [Connector('2', float)])

    def step(self, node):
        node.setOut('2', 2.0)

class E(SimSchema):
    '''Emits Euler's constant.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'E', [], [Connector('e', float)])

    def step(self, node):
        node.setOut('e', math.e)

class Pi(SimSchema):
    '''Emits pi, the ratio of a circle's circumference to its diameter.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'PI', [], [Connector('p', float)])

    def step(self, node):
        node.setOut('p', math.pi)

class Tau(SimSchema):
    '''Emits tau, the ratio of a circle's circumference to its radius.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'TAU', [], [Connector('t', float)])

    def step(self, node):
        node.setOut('t', 2*math.pi)

class Infinity(SimSchema):
    '''Emits the floating point representation of positive infinity.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', '+INF', [], [Connector('Q', float)])

    def step(self, node):
        node.setOut('Q', float('inf'))

class NegativeInfinity(SimSchema):
    '''Emits the floating point representation of negative infinity.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', '-INF', [], [Connector('Q', float)])

    def step(self, node):
        node.setOut('Q', float('-inf'))

class FloatConst(SimSchema):
    '''Emits the floating point constant set as its `value' property.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'FCON', [], [Connector('Q', float)], [Property('value', float, 0.0)])

    def step(self, node):
        node.setOut('Q', node.getProp('value'))

class IntConst(SimSchema):
    '''Emits the integer constant set as its `value' property.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'ICON', [], [Connector('Q', int)], [Property('value', int, 0)])

    def step(self, node):
        node.setOut('Q', node.getProp('value'))

class StringConst(SimSchema):
    '''Emits the string constant set as its `value' property.'''
    def __init__(self):
        SimSchema.__init__(self, 'Constants', 'SCON', [], [Connector('Q', str)], [Property('value', str, '')])

    def step(self, node):
        node.setOut('Q', node.getProp('value'))

################################################################################

class Equal(SimSchema):
    '''Computes the boolean truth of A == B.'''
    def __init__(self):
        SimSchema.__init__(self, 'Comparison', 'EQ', [
            Connector('A'),
            Connector('B'),
        ], [
            Connector('Q', bool),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') == node.getIn('B'))

class Less(SimSchema):
    '''Computes the boolean truth of A < B. The logical negation of this value is the truth of A >= B'''
    def __init__(self):
        SimSchema.__init__(self, 'Comparison', 'LS', [
            Connector('A'),
            Connector('B'),
        ], [
            Connector('Q', bool),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') < node.getIn('B'))

class Greater(SimSchema):
    '''Computes the boolean truth of A > B. The logical negation of this value is the truth of A <= B'''
    def __init__(self):
        SimSchema.__init__(self, 'Comparison', 'GT', [
            Connector('A'),
            Connector('B'),
        ], [
            Connector('Q', bool),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') > node.getIn('B'))

################################################################################

class And(SimSchema):
    '''Computes the logical conjunction of all of its inputs. For no inputs, emits True.'''
    def __init__(self):
        SimSchema.__init__(self, 'Logic', 'AND', [
            Connector('I', bool, False),
        ], [
            Connector('Q', bool),
        ])

    def step(self, node):
        node.setOut('Q', all(node.getIns('I')))

class Or(SimSchema):
    '''Computes the logical disjunction of all of its inputs. For no inputs, emits False.'''
    def __init__(self):
        SimSchema.__init__(self, 'Logic', 'OR', [
            Connector('I', bool, False),
        ], [
            Connector('Q', bool),
        ])

    def step(self, node):
        node.setOut('Q', any(node.getIns('I')))

class Not(SimSchema):
    '''Computes the logical negation of its input.'''
    def __init__(self):
        SimSchema.__init__(self, 'Logic', 'NOT', [
            Connector('I', bool),
        ], [
            Connector('Q', bool),
        ])

    def step(self, node):
        node.setOut('Q', not node.getIn('I'))

################################################################################

class ToInteger(SimSchema):
    '''Casts input I to an integer.'''
    def __init__(self):
        SimSchema.__init__(self, 'Conversion', '2INT', [
            Connector('I'),
        ], [
            Connector('Q', int),
        ])

    def step(self, node):
        node.setOut('Q', int(node.getIn('I')))

class ToString(SimSchema):
    '''Casts input I to a string.'''
    def __init__(self):
        SimSchema.__init__(self, 'Conversion', '2STR', [
            Connector('I'),
        ], [
            Connector('Q', str),
        ])

    def step(self, node):
        node.setOut('Q', str(node.getIn('I')))

################################################################################

class FileWrite(SimSchema):
    '''Writes to the beginning of, or append to, `file' the string value of I on each tick, depending on the value of `append', and closes the file if `close' is a boolean truth.'''
    def __init__(self):
        SimSchema.__init__(self, 'Files', 'FIWR', [
            Connector('I'),
        ], [], [
            Property('file', str, ''),
            Property('append', int, 0),
            Property('close', int, 0),
        ])

    def reset(self, node):
        node.fp = open(node.getProp('file'), 'a' if node.getProp('append') else 'w')

    def step(self, node):
        if node.getProp('close'):
            self.reset(node)
        if not node.getProp('append'):
            try:
                node.fp.seek(0, 0)
            except OSError:
                node.fp = open(node.getProp('file'), 'w')
        node.fp.write(str(node.getIn('I')))
        if node.getProp('close'):
            node.fp.close()

################################################################################

class Buffer(SimSchema):
    '''Emits its input, delayed by one simulation tick. This may be used in the default stepper (ParallelStepper) to synchronize inputs, and is otherwise useless. DelayBuffer provides more flexibility.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'BUF', [
            Connector('I'),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('I'))

class DelayBuffer(SimSchema):
    '''Emits its input, delayed by `delay' simulation ticks. This may be used in the default stepper (ParallelStepper) to synchronize inputs.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'DLAY', [
            Connector('I'),
        ], [
            Connector('Q'),
        ], [
            Property('delay', int, 1, min=1)
        ])

    def step(self, node):
        if node.getProp('delay') <= 1:
            node.setOut('Q', node.getIn('I'))
            return
        if not hasattr(node, 'delay_buffer'):
            node.delay_buffer = [None] * (node.getProp('delay') - 1)
            node.delay_index = 0
        node.delay_buffer[node.delay_index] = node.getIn('I')
        node.delay_index += 1
        if node.delay_index >= len(node.delay_buffer):
            node.delay_index = 0
        node.setOut('Q', node.delay_buffer[node.delay_index])
        node.delay_index += 1
        if node.delay_index >= len(node.delay_buffer):
            node.delay_index = 0

class Printer(SimSchema):
    '''Prints the list consisting of all of its input values to stdout.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'PRINT', [
            Connector('I', None, False),
        ], [])

    def step(self, node):
        print(node.getIns('I'))

class SetBus(SimSchema):
    '''Sets the simulator bus named `bus' to the input value.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'SBUS', [Connector('I')], [], [Property('bus', str, '')])

    def step(self, node):
        node.graph.setBus(node.getProp('bus'), node.getIn('I'))

class GetBus(SimSchema):
    '''Gets the value of the simulator bus named `bus'.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'GBUS', [], [Connector('Q')], [Property('bus', str, '')])

    def step(self, node):
        node.setOut('Q', node.graph.getBus(node.getProp('bus')))

class Python(SimSchema):
    '''Evaluates `expr' as Python code, and emits the result to Q. `invars' is a comma-separated list of additional inputs to this node whose values will be available as variables of the same name in the expression. `invars' should not contain whitespace or other characters illegal in identifiers.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'PY', [], [Connector('Q')], [
            Property('expr', str, ''),
            Property('invars', str, ''),
        ])

    def getIns(self, node):
        return [Connector(name) for name in node.getProp('invars', '').split(',')]

    def getInMap(self, node):
        return {con.name: con for con in self.getIns(node)}

    def step(self, node):
        ins = {name: node.getIn(name) for name in node.getProp('invars', '').split(',')}
        node.setOut('Q', eval(node.getProp('expr'), globals(), ins))

class Default(SimSchema):
    '''Emits its input I, so long as it is not the null value (None); otherwise, emits D.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'DFLT', [
            Connector('I'),
            Connector('D'),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        inval = node.getIn('I')
        if inval is None:
            node.setOut('Q', node.getIn('D'))
        else:
            node.setOut('Q', inval)

class Null(SimSchema):
    '''Emits the null value (None). Note that this is equivalent to leaving an input disconnected.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'NULL', [], [Connector('Q')])

    def step(self, node):
        pass

class Select(SimSchema):
    '''If C is a boolean truth, emit T. Otherwise, emit F.'''
    def __init__(self):
        SimSchema.__init__(self, 'Generic', 'SLCT', [
            Connector('T'),
            Connector('F'),
            Connector('C', bool),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        cond = node.getIn('C')
        if cond:
            node.setOut('Q', node.getIn('T'))
        else:
            node.setOut('Q', node.getIn('F'))

################################################################################

class Sine(SimSchema):
    '''Computes the sine in radians of I.'''
    def __init__(self):
        SimSchema.__init__(self, 'Trigonometry', 'SIN', [
            Connector('I', float),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', math.sin(node.getIn('I')))

class Cosine(SimSchema):
    '''Computes the cosine in radians of I.'''
    def __init__(self):
        SimSchema.__init__(self, 'Trigonometry', 'COS', [
            Connector('I', float),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', math.cos(node.getIn('I')))

class Tangent(SimSchema):
    '''Computes the tangent in radians of I.'''
    def __init__(self):
        SimSchema.__init__(self, 'Trigonometry', 'TAN', [
            Connector('I', float),
        ], [
            Connector('Q', float),
        ])

    def step(self, node):
        node.setOut('Q', math.tan(node.getIn('I')))

################################################################################

class Ticks(SimSchema):
    '''Returns the number of ticks this simulation has run for, starting at 0 on the first run or at the beginning of an aggregate subsimulation.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'TICK', [], [Connector('Q', int)])

    def step(self, node):
        node.setOut('Q', node.graph.tick)

class Time(SimSchema):
    '''Returns real time in seconds since the epoch.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'TIME', [], [Connector('Q', float)])

    def step(self, node):
        node.setOut('Q', time.time())

class PerformanceCounter(SimSchema):
    '''Returns a high-precision counter in seconds with an arbitrary epoch. The resolution is fine enough that "parallel" invocations of independent nodes will have (subtle) different values.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'PCNT', [], [Connector('Q', float)])

    def step(self, node):
        node.setOut('Q', time.perf_counter())

class ProcessClock(SimSchema):
    '''Returns a counter in seconds whose epoch is the start of the process. The resolution is in units used by the operating system's process scheduler. Note that simulation may not necessarily start at 0 seconds, due to setup or simulation in another process.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'PCLK', [], [Connector('Q', float)])

    def step(self, node):
        node.setOut('Q', time.clock())

class Start(SimSchema):
    '''Returns the real time (as with Time) that this simulation was last reset (i.e., when its tick count was zeroed).'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'START', [], [Connector('Q', float)])

    def step(self, node):
        node.setOut('Q', node.graph.starttime)

class Integral(SimSchema):
    '''Maintains an internal value (initially `init') and changes the value every tick by d, which is in units per second. If R is a boolean truth, this internal value is unconditionally set to V. Emits the internal value.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'INTG', [
            Connector('d', float),
            Connector('V', float),
            Connector('R', bool),
        ], [
            Connector('Q', float),
        ], [
            Property('init', float, 0.0),
        ])

    def reset(self, node):
        node.value = node.getProp('init')
        node.lasttime = time.perf_counter()

    def step(self, node):
        if node.getIn('R'):
            node.value = node.getIn('V')
        else:
            node.value += node.getIn('d') * (time.perf_counter() - node.lasttime)
        node.setOut('Q', node.value)
        node.lasttime = time.perf_counter()

class Accumulator(SimSchema):
    '''Maintains an internal value (initially `init') and changes the value every tick by d. If R is a boolean truth, this internal value is unconditionally set to V. Emits the internal value.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'ACCM', [
            Connector('d', float),
            Connector('V', float),
            Connector('R', bool),
        ], [
            Connector('Q', float),
        ], [
            Property('init', float, 0.0),
        ])

    def reset(self, node):
        node.value = node.getProp('init')

    def step(self, node):
        if node.getIn('R'):
            node.value = node.getIn('V')
        else:
            node.value += node.getIn('d')
        node.setOut('Q', node.value)

class Derivative(SimSchema):
    '''Takes the delta (as with Delta) between I and its previous value (initially `init'), and divides it by the change in time, giving a result in units per second.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'DERV', [
            Connector('I', float),
        ], [
            Connector('Q', float),
        ], [
            Property('init', float, 0.0),
        ])

    def reset(self, node):
        node.value = node.getProp('init')
        node.lasttime = time.perf_counter()

    def step(self, node):
        value = node.getIn('I')
        node.setOut('Q', (value - node.value) / (time.perf_counter() - node.lasttime))
        node.value = value
        node.lasttime = time.perf_counter()

class Delta(SimSchema):
    '''Takes the difference between I and its value on the previous tick (initially `init').'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'DELT', [
            Connector('I', float),
        ], [
            Connector('Q', float),
        ], [
            Property('init', float, 0.0)
        ])

    def reset(self, node):
        node.value = node.getProp('init')

    def step(self, node):
        value = node.getIn('I')
        delta = value - node.value
        node.value = value
        node.setOut('Q', delta)

class RunningMaximum(SimSchema):
    '''Maintains the running maximum of its input values over time. Initially, or if R is a boolean truth, it resets immediately to the value of its input.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'RMAX', [
            Connector('I', float),
            Connector('R', bool),
        ], [
            Connector('Q', float),
        ])

    def reset(self, node):
        node.value = None

    def step(self, node):
        if node.getIn('R') or node.value is None:
            node.value = node.getIn('I')
        else:
            node.value = max(node.value, node.getIn('I'))
        node.setOut('Q', node.value)

class RunningMinimum(SimSchema):
    '''Maintains the running minimum of its input values over time. Initially, or if R is a boolean truth, it resets immediately to the value of its input.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'RMIN', [
            Connector('I', float),
            Connector('R', bool),
        ], [
            Connector('Q', float),
        ])

    def reset(self, node):
        node.value = None

    def step(self, node):
        if node.getIn('R') or node.value is None:
            node.value = node.getIn('I')
        else:
            node.value = min(node.value, node.getIn('I'))
        node.setOut('Q', node.value)

class RunningAverage(SimSchema):
    '''Maintains the running average of its input values over time. Initially, or if R is a boolean truth, it resets immediately to the value of its input.'''
    def __init__(self):
        SimSchema.__init__(self, 'Time', 'RAVG', [
            Connector('I', float),
            Connector('R', bool),
        ], [
            Connector('Q', float),
        ])

    def reset(self, node):
        node.value = None
        node.counter = 0

    def step(self, node):
        if node.getIn('R') or node.value is None:
            node.value = node.getIn('I')
            node.counter = 1
        else:
            node.counter += 1
            node.value = ((node.counter - 1) * node.value + node.getIn('I')) / node.counter
        node.setOut('Q', node.value)

################################################################################

class ListGetIndex(SimSchema):
    '''Emits the element at index I of list L. If I is out of range, emits D instead. Non-integral values of I (including null) will emit the first element of the list.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'LGIX', [
            Connector('L', list),
            Connector('I', int),
            Connector('D'),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        idx = node.getIn('I')
        lst = node.getIn('L')
        if idx >= 0 and idx < len(lst):
            node.setOut('Q', lst[idx])
        else:
            node.setOut('Q', node.getIn('D'))

class ListSetIndex(SimSchema):
    '''Emits a list which is the same as L, but with index I set to E. If I is out of range, emits L. Non-integral values of I (including null) set the first element of the list.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'LSIX', [
            Connector('L', list),
            Connector('I', int),
            Connector('E'),
        ], [
            Connector('Q', list)
        ])

    def step(self, node):
        idx = node.getIn('I')
        l = node.getIn('L')[:]
        if idx >= 0 and idx < len(l):
            l[idx] = node.getIn('E')
        node.setOut('Q', l)

class ListAppend(SimSchema):
    '''Emits a list consisting of L suffixed with an element E.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'LAPP', [
            Connector('L', list),
            Connector('E'),
        ], [
            Connector('Q', list)
        ])

    def step(self, node):
        l = node.getIn('L')[:]
        l.append(node.getIn('E'))
        node.setOut('Q', l)

class ListInsert(SimSchema):
    '''Emits a list consisting of L with an element E at index I displacing elements in L to I + 1.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'LINS', [
            Connector('L', list),
            Connector('I', int),
            Connector('E'),
        ], [
            Connector('Q', list)
        ])

    def step(self, node):
        l = node.getIn('L')[:]
        l.insert(node.getIn('I'), node.getIn('E'))
        node.setOut('Q', l)

class ListSlice(SimSchema):
    '''Emits a list consisting of a subsequence of L starting at and including index B and ending at but not including index U. Negative values are normalized by adding the length of L to them (so -1 is the end of the list, -2 is one before that, etc.). After normalization, invalid values are clamped to 0 for B and one less than the length of L for U, so the resulting sequence will never be longer than L. Non-integral values (including null) are interpreted as zero.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'LSLC', [
            Connector('L', list),
            Connector('B', int),
            Connector('U', int),
        ], [
            Connector('Q', list),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('L')[node.getIn('B'):node.getIn('U')])

class ListRepeat(SimSchema):
    '''Emits a list consisting of the sequence in L repeated R times. Non-integral values of R are interpreted as zero; the result will be an empty list.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'LRPT', [
            Connector('L', list),
            Connector('R', int),
        ], [
            Connector('Q', list)
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('L') * node.getIn('R'))

class ListConcat(SimSchema):
    '''Emits a list consisting of the elements of A concatenated with the elements of B--the length is the sum of the lengths of its inputs.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'LCCT', [
            Connector('A', list),
            Connector('B', list),
        ], [
            Connector('Q', list),
        ])

    def step(self, node):
        node.setOut('Q', node.getIn('A') + node.getIn('B'))

class EmptyList(SimSchema):
    '''Emits an empty list.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'ELST', [], [Connector('Q', list)])

    def step(self, node):
        node.setOut('Q', [])

class InputList(SimSchema):
    '''Emits a list consisting of all values to polyinput I. There is no way to influence the order, but it will remain consistent within a simulation.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'ILST', [
            Connector('I', None, False),
        ], [
            Connector('Q', list),
        ])

    def step(self, node):
        node.setOut('Q', node.getIns('I'))

class Sum(SimSchema):
    '''Computes the sum of all values in a list. An empty sum will be 0.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'SUM', [
            Connector('I', list),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        node.setOut('Q', sum(node.getIn('I')))

class Product(SimSchema):
    '''Computes the product of all values in a list. An empty product will be 1.0.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'PROD', [
            Connector('I', list),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        node.setOut('Q', functools.reduce(operator.mul, node.getIn('I'), 1.0))

class Maximum(SimSchema):
    '''Computes the maximum of all values in a list. An empty maximum results in a null value.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'MAX', [
            Connector('I', list),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        node.setOut('Q', max(node.getIn('I')))

class Minimum(SimSchema):
    '''Computes the minimum of all values in a list. An empty minimum results in a null value.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'MIN', [
            Connector('I', list),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        node.setOut('Q', min(node.getIn('I')))

class Length(SimSchema):
    '''Emits the length of the input list.'''
    def __init__(self):
        SimSchema.__init__(self, 'Sequences', 'LEN', [
            Connector('I', list),
        ], [
            Connector('Q', int),
        ])

    def step(self, node):
        node.setOut('Q', len(node.getIn('I')))

################################################################################

class MapGet(SimSchema):
    '''Computes the value V corresponding to the key K in mapping M. If K is not in M, emits D instead.'''
    def __init__(self):
        SimSchema.__init__(self, 'Mappings', 'MGET', [
            Connector('M', dict),
            Connector('K'),
            Connector('D'),
        ], [
            Connector('V')
        ])

    def step(self, node):
        node.setOut('V', node.getIn('M').get(node.getIn('K'), node.getIn('D')))

class MapSet(SimSchema):
    '''Emits a map containing all items in M, plus an association with key K and value V. If K was in M, that association is lost.'''
    def __init__(self):
        SimSchema.__init__(self, 'Mappings', 'MSET', [
            Connector('M', dict),
            Connector('K'),
            Connector('V'),
        ], [
            Connector('Q'),
        ])

    def step(self, node):
        d = node.getIn('M')
        d.update({node.getIn('K'): node.getIn('V')})
        node.setOut('Q', d)

class MapKeys(SimSchema):
    '''Computes the list of all keys of M in no particular order.'''
    def __init__(self):
        SimSchema.__init__(self, 'Mappings', 'MKYS', [
            Connector('M', dict),
        ], [
            Connector('Q', list),
        ])

    def step(self, node):
        node.setOut('Q', list(node.getIn('M').keys()))

class MapValues(SimSchema):
    '''Computes the list of all values of M in no particular order.'''
    def __init__(self):
        SimSchema.__init__(self, 'Mappings', 'MVLS', [
            Connector('M', dict),
        ], [
            Connector('Q', list),
        ])

    def step(self, node):
        node.setOut('Q', list(node.getIn('M').values()))

class MapItems(SimSchema):
    '''Computes a list containing all keys and values as sequences of length 2 in that order, themselves in no particular order.'''
    def __init__(self):
        SimSchema.__init__(self, 'Mappings', 'MITS', [
            Connector('M', dict),
        ], [
            Connector('Q', list),
        ])

    def step(self, node):
        node.setOut('Q', list(node.getIn('M').items()))

class EmptyMap(SimSchema):
    '''Emits an empty map.'''
    def __init__(self):
        SimSchema.__init__(self, 'Mappings', 'EMAP', [], [Connector('Q', dict)])

    def step(self, node):
        node.setOut('Q', {})

class MapFromItems(SimSchema):
    '''Construct a map from a list containing sequences of length 2 representing a key and an associated value.'''
    def __init__(self):
        SimSchema.__init__(self, 'Mappings', 'MFIT', [
            Connector('L', list),
        ], [
            Connector('Q', dict),
        ])

    def step(self):
        node.setOut('Q', dict(node.getIn('L')))

################################################################################

class Subsimulator(SimSchema):
    '''Loads a simulation graph `simfile' relative to the current working directory; the inputs and outputs are named in the comma-separated lists `inbusses' and `outbusses', respectively. Included whitespace will be made part of the bus names, and should be avoided. Resets the subsimulation when the simulation resets. On each tick, loads the input values into the busses of the subsimulation, runs it by one tick, and emits the named busses to the outputs.'''
    def __init__(self):
        SimSchema.__init__(self, 'Subsimulation', 'SSIM', [], [], [
            Property('simfile', str, ''),
            Property('inbusses', str, ''),
            Property('outbusses', str, ''),
        ])

    def getIns(self, node):
        return [Connector(name) for name in node.getProp('inbusses', '').split(',')]

    def getInMap(self, node):
        return {con.name: con for con in self.getIns(node)}

    def getOuts(self, node):
        return [Connector(name) for name in node.getProp('outbusses', '').split(',')]

    def getOutMap(self, node):
        return {con.name: con for con in self.getOuts(node)}

    def reset(self, node):
        node.subsim = Simulator()
        tree = ET.parse(node.getProp('simfile'))
        node.subsim.restore(tree, SchemaRegistry.ALL, ConsumerRegistry.ALL, ProducerRegistry.ALL)
        node.subsim.reset()

    def step(self, node):
        for ibus in node.getProp('inbusses').split(','):
            node.subsim.bus[ibus] = node.getIn(ibus)
        node.subsim.step()
        node.subsim.advance()
        for obus in node.getProp('outbusses').split(','):
            node.setOut(obus, node.subsim.bus[obus])

class SubsimMap(SimSchema):
    '''Loads a simulation graph `simfile' relative to the current working directory. On each tick, accepts a list I, and the inputs given by the comma-separated list in `inbusses', and resets the subsimulation if `untilset' is 1; for each element of I, sets the bus named `inbus' to the element, `indexbus' to the index of the element, `listbus' to I, and each listed inbus to the value on the respective input; then runs the subsimulation for one tick if `untilset' is 0, or until a non-null value is produces on the `outbus' in up to `ticklimit' ticks, after at least `minticks' ticks have passed. At this point, the value produced on the subsimulation bus `outbus' is saved in the output list Q at the same index. Whitespace in `inbusses' will be included in bus names, and is not recommended. Q's length is always the same as I's.'''
    def __init__(self):
        SimSchema.__init__(self, 'Subsimulation', 'SMAP', [
            Connector('I', list)
        ], [
            Connector('Q', list),
        ], [
            Property('simfile', str, ''),
            Property('inbus', str, ''),
            Property('indexbus', str, ''),
            Property('listbus', str, ''),
            Property('outbus', str, ''),
            Property('untilset', int, 1, min=0, max=1),
            Property('ticklimit', int, 25, min=1),
            Property('minticks', int, 0, min=0),
            Property('inbusses', str, ''),
        ])

    def getIns(self, node):
        return self.ins + ([Connector(name) for name in node.getProp('inbusses').split(',')] if node.getProp('inbusses') else [])

    def getInMap(self, node):
        return {con.name: con for con in self.getIns(node)}

    def reset(self, node):
        node.subsim = Simulator()
        tree = ET.parse(node.getProp('simfile'))
        node.subsim.restore(tree, SchemaRegistry.ALL, ConsumerRegistry.ALL, ProducerRegistry.ALL)
        node.subsim.reset()

    def step(self, node):
        ibus, obus, lbus, xbus = node.getProp('inbus'), node.getProp('outbus'), node.getProp('listbus'), node.getProp('indexbus')
        ticklim = node.getProp('ticklimit')
        minticks = node.getProp('minticks')
        untilset = node.getProp('untilset')
        inbusses = node.getProp('inbusses')
        ins = node.getIn('I')
        outs = [None] * len(ins)
        for idx, inval in enumerate(ins):
            if untilset:
                node.subsim.reset()
            node.subsim.bus[ibus] = inval
            node.subsim.bus[xbus] = idx
            node.subsim.bus[lbus] = ins
            if inbusses:
                for bus in inbusses.split(','):
                    node.subsim.bus[bus] = node.getIn(bus)
            node.subsim.bus[obus] = None
            if untilset:
                while node.subsim.tick < minticks or (node.subsim.bus[obus] is None and node.subsim.tick < ticklim):
                    node.subsim.step()
                    node.subsim.advance()
                    node.subsim.tick += 1
            else:
                node.subsim.step()
                node.subsim.advance()
                node.subsim.tick += 1
            outs[idx] = node.subsim.bus[obus]
        node.setOut('Q', outs)

class SubsimFilter(SimSchema):
    '''Loads a simulation graph `simfile' relative to the current working directory. On each tick, accepts a list I, and the inputs given by the comma-separated list in `inbusses', and resets the subsimulation if `untilset' is 1; for each element of I, sets the bus named `inbus' to the element, `indexbus' to the index of the element, `listbus' to I, and each listed inbus to the value on the respective input; then runs the subsimulation for one tick if `untilset' is 0, or until a non-null value is produces on the `filterbus' in up to `ticklimit' ticks, after at least `minticks' ticks have passed. At this point, the value produced on the subsimulation bus `filterbus' determines whether or not the corresponding element of I is included in Q. Whitespace in `inbusses' will be included in bus names, and is not recommended. Q's length is always less than or equal to I's, and the relative ordering of elements is preserved.'''
    def __init__(self):
        SimSchema.__init__(self, 'Subsimulation', 'SFLT', [
            Connector('I', list)
        ], [
            Connector('Q', list),
        ], [
            Property('simfile', str, ''),
            Property('inbus', str, ''),
            Property('indexbus', str, ''),
            Property('listbus', str, ''),
            Property('filterbus', str, ''),
            Property('untilset', int, 1, min=0, max=1),
            Property('ticklimit', int, 250, min=1),
            Property('minticks', int, 0, min=0),
            Property('inbusses', str, ''),
        ])

    def getIns(self, node):
        return self.ins + ([Connector(name) for name in node.getProp('inbusses').split(',')] if node.getProp('inbusses') else [])

    def getInMap(self, node):
        return {con.name: con for con in self.getIns(node)}

    def reset(self, node):
        node.subsim = Simulator()
        tree = ET.parse(node.getProp('simfile'))
        node.subsim.restore(tree, SchemaRegistry.ALL, ConsumerRegistry.ALL, ProducerRegistry.ALL)
        node.subsim.reset()

    def step(self, node):
        ibus, fbus, lbus, xbus = node.getProp('inbus'), node.getProp('filterbus'), node.getProp('listbus'), node.getProp('indexbus')
        ticklim = node.getProp('ticklimit')
        minticks = node.getProp('minticks')
        untilset = node.getProp('untilset')
        inbusses = node.getProp('inbusses')
        ins = node.getIn('I')
        outs = []
        for idx, inval in enumerate(ins):
            if untilset:
                node.subsim.reset()
            node.subsim.bus[ibus] = inval
            node.subsim.bus[xbus] = idx
            node.subsim.bus[lbus] = ins
            if inbusses:
                for bus in inbusses.split(','):
                    node.subsim.bus[bus] = node.getIn(bus)
            node.subsim.bus[fbus] = None
            if untilset:
                while node.subsim.tick < minticks or (node.subsim.bus[fbus] is None and node.subsim.tick < ticklim):
                    #print('!!!!! index {} step {} !!!!!'.format(idx, node.subsim.tick))
                    node.subsim.step()
                    node.subsim.advance()
                    node.subsim.tick += 1
            else:
                node.subsim.step()
                node.subsim.advance()
                node.subsim.tick += 1
            #print('!!!!! index {} fbus result {} on tick {} !!!!!'.format(idx, node.subsim.bus[fbus], node.subsim.tick))
            if node.subsim.bus[fbus]:
                outs.append(inval)
        node.setOut('Q', outs)

class SubsimReduce(SimSchema):
    '''Loads a simulation graph `simfile' relative to the current working directory. On each tick, accepts a list I and an initial value S, and the inputs given by the comma-separated list in `inbusses', and resets the subsimulation if `untilset' is 1; for each element of I, sets the bus named `abus' to the current value, `bbus' to the current element, `indexbus' to the index of the element, `listbus' to I, and each listed inbus to the value on the respective input; then runs the subsimulation for one tick if `untilset' is 0, or until a non-null value is produces on the `outbus' in up to `ticklimit' ticks, after at least `minticks' ticks have passed. At this point, the value produced on the subsimulation bus `outbus' becomes the current value. Whitespace in `inbusses' will be included in bus names, and is not recommended. Q is the scalar result of the current value at the end of the simulation.'''
    def __init__(self):
        SimSchema.__init__(self, 'Subsimulation', 'SRED', [
            Connector('I', list),
            Connector('S'),
        ], [
            Connector('Q'),
        ], [
            Property('simfile', str, ''),
            Property('abus', str, ''),
            Property('bbus', str, ''),
            Property('indexbus', str, ''),
            Property('listbus', str, ''),
            Property('outbus', str, ''),
            Property('untilset', int, 1, min=0, max=1),
            Property('ticklimit', int, 250, min=1),
            Property('minticks', int, 0, min=0),
            Property('inbusses', str, ''),
        ])

    def getIns(self, node):
        return self.ins + ([Connector(name) for name in node.getProp('inbusses').split(',')] if node.getProp('inbusses') else [])

    def getInMap(self, node):
        return {con.name: con for con in self.getIns(node)}

    def reset(self, node):
        node.subsim = Simulator()
        tree = ET.parse(node.getProp('simfile'))
        node.subsim.restore(tree, SchemaRegistry.ALL, ConsumerRegistry.ALL, ProducerRegistry.ALL)
        node.subsim.reset()

    def step(self, node):
        abus, bbus, obus, lbus, xbus = node.getProp('abus'), node.getProp('bbus'),  node.getProp('outbus'), node.getProp('listbus'), node.getProp('indexbus')
        ticklim = node.getProp('ticklimit')
        minticks = node.getProp('minticks')
        untilset = node.getProp('untilset')
        inbusses = node.getProp('inbusses')
        ins = node.getIn('I')
        value = node.getIn('S')
        for idx, inval in enumerate(ins):
            if untilset:
                node.subsim.reset()
            node.subsim.bus[abus] = value
            node.subsim.bus[bbus] = inval
            node.subsim.bus[xbus] = idx
            node.subsim.bus[lbus] = ins
            if inbusses:
                for bus in inbusses.split(','):
                    node.subsim.bus[bus] = node.getIn(bus)
            node.subsim.bus[obus] = None
            if untilset:
                while node.subsim.tick < minticks or (node.subsim.bus[obus] is None and node.subsim.tick < ticklim):
                    #print('!!!!! index {} step {} !!!!!'.format(idx, node.subsim.tick))
                    node.subsim.step()
                    node.subsim.advance()
                    node.subsim.tick += 1
            else:
                node.subsim.step()
                node.subsim.advance()
                node.subsim.tick += 1
            #print('!!!!! index {} fbus result {} on tick {} !!!!!'.format(idx, node.subsim.bus[fbus], node.subsim.tick))
            value = node.subsim.bus[obus]
        node.setOut('Q', value)

################################################################################

try:
    import numpy
except ImportError:
    pass
else:
    from lib_numpy import *
    try:
        import scipy
    except ImportError:
        pass
    else:
        from lib_scipy import *
