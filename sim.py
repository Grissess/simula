import traceback, sys, time, functools, multiprocessing
import xml.etree.ElementTree as ET

from model import *

class Property(object):
    def __init__(self, name, type=None, default=None, **kwargs):
        self.name = name
        self.type = type
        self.default = default
        for k, v in kwargs.items():
            setattr(self, k, v)

class Monitor(object):
    def onStepError(self, node, err):
        sys.stderr.write('===== Exception on tick {} stepping {} on {} =====\n'.format(node.graph.tick, node.schema, self))
        traceback.print_exc()
        sys.stderr.write('----- Parameters: -----\n')
        for con in node.schema.getIns(node):
            sys.stderr.write('{}: {}\n'.format(con.name, node.getIns(con.name)))
        sys.stderr.write('----- Busses: -----\n')
        for k, v in node.graph.bus.items():
            sys.stderr.write('{}: {}\n'.format(k, v))
        sys.stderr.write('='*80 + '\n')

    def onConvError(self, node, in_, value, tp, e):
        sys.stderr.write('===== Exception in conversion from {} to {} on {} of {}, using default =====\n'.format(value, tp, node.schema, node))

class SNode(Node):
    SPECIAL_DEFAULTS = {}
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        self.values = {}
        self.newvalues = {}
        self.props = {}
        for prop in self.schema.props:
            if prop.default is not None:
                self.props[prop.name] = prop.default

    def reset(self):
        self.cins = self.schema.getIns(self)
        self.couts = self.schema.getOuts(self)
        for con in self.couts:
            self.values[con.name] = None
            self.newvalues[con.name] = None
        self.schema.reset(self)
        self.typecache = {}

    def step(self):
        try:
            self.schema.step(self)
        except Exception as e:
            self.graph.monitor.onStepError(self, e)

    def advance(self):
        for con in self.couts:
            self.values[con.name] = self.newvalues[con.name]
            self.newvalues[con.name] = None

    def getIns(self, name):
        edges = self.incoming[name]
        if not edges:
            return []
        values = [node.values.get(out) for node, out in edges]
        tp = self.typecache.get(name)
        if not tp:
            con = self.schema.getInMap(self).get(name)
            if con:
                tp = con.type
                self.typecache[name] = tp
        if tp:
            for idx, val in enumerate(values):
                if isinstance(val, tp):
                    continue
                try:
                    values[idx] = tp(val)
                except (ValueError, TypeError) as e:
                    self.graph.monitor.onConvError(self, name, val, tp, e)
                    if tp in self.SPECIAL_DEFAULTS:
                        values[idx] = self.SPECIAL_DEFAULTS[tp](tp)
                    else:
                        values[idx] = tp()
        return values

    def getIn(self, name):
        ins = self.getIns(name)
        try:
            return ins[0]
        except IndexError:
            return None

    def getProp(self, name, default=None):
        return self.props.get(name, default)

    def setOut(self, name, value):
        self.newvalues[name] = value

    def save(self, root):
        this = Node.save(self, root)
        xprops = ET.SubElement(this, 'props')
        for prop, val in self.props.items():
            xprop = ET.SubElement(xprops, 'prop', name=prop, val=str(val))
        return this

    def restore(self, this, idmap):
        xprops = this.find('props')
        if xprops is not None:
            for xprop in xprops.findall('prop'):
                prop = self.schema.propmap.get(xprop.get('name'))
                if prop:
                    self.props[prop.name] = prop.type(xprop.get('val'))
        Node.restore(self, this, idmap)

try:
    import numpy as np
except ImportError:
    pass
else:
    EMPTY_FLOAT_ARRAY = np.ndarray(0, dtype=np.float32)
    SNode.SPECIAL_DEFAULTS[np.ndarray] = lambda tp: EMPTY_FLOAT_ARRAY

class BaseProducer(object):
    PROPS = [Property('bus', str, '')]

    def reset(self, sim):
        pass

    def step(self, sim):
        raise NotImplementedError('BaseProducer.step()')

    def advance(self, sim):
        pass

    def stepsNeeded(self, sim):
        return 1

    def delayNeeded(self, sim):
        return None

class BaseConsumer(object):
    PROPS = [Property('bus', str, '')]

    def reset(self, sim):
        pass

    def step(self, sim):
        raise NotImplementedError('BaseConsumer.step()')

    def advance(self, sim):
        pass

    def stepsNeeded(self, sim):
        return 1

    def delayNeeded(self, sim):
        return None

class StepperRegistry(type):
    ALL = {}
    def __new__(mcs, name, bases, dict):
        tp = type.__new__(mcs, name, bases, dict)
        if name != 'Stepper':
            inst = tp()
            mcs.ALL[name] = inst
        return tp

class Stepper(object, metaclass=StepperRegistry):
    def reset(self, sim):
        print('-- Stepper %r resets simulation %r --'%(self, sim))
        for node in sim.nodes:
            node.reset()

    def step(self, sim):
        raise NotImplementedError('Stepper.step()')

    def advance(self, sim):
        for node in sim.nodes:
            node.advance()

class ParallelStepper(Stepper):
    def step(self, sim):
        for node in sim.nodes:
            node.step()

class TopologicalStepper(Stepper):
    def reset(self, sim):
        Stepper.reset(self, sim)
        if not hasattr(sim, 'order'):
            sim.order = []
            deps = {node: functools.reduce(set.union, (set((i[0] for i in pairs)) for pairs in node.incoming.values()), set()) for node in sim.nodes}
            while True:
                ordered = {node for node, dep in deps.items() if not dep}
                if not ordered:
                    if not deps:
                        break
                    ordered = {min(((node, len(dep)) for node, dep in deps.items()), key=lambda pair: pair[1])[0]}
                sim.order.extend(ordered)
                deps = {node: (dep - ordered) for node, dep in deps.items() if node not in ordered}
            print('----- Ordering -----')
            for node in sim.order:
                print(' - {} of {}'.format(node.schema, node))
            print('-'*80)
            assert not deps

    def step(self, sim):
        for node in sim.order:
            node.step()
            node.advance()

    def advance(self, sim):
        pass

class StepProcess(multiprocessing.Process):
    def run(self):
        nodes = self._args[0]
        while True:
            for node in nodes:
                node.step()
            for node in nodes:
                node.advance()

class MulticoreStepper(Stepper):
    def reset(self, sim):
        Stepper.reset(self, sim)
        self.pools = [[] for i in range(multiprocessing.cpu_count())]
        i = 0
        for node in sim.nodes:
            self.pools[i].append(node)
            i += 1
            if i >= len(self.pools):
                i = 0
        self.processes = [StepProcess(args=(pool,)) for pool in self.pools]
        self.started = False

    def step(self, sim):
        if not self.started:
            for proc in self.processes:
                proc.start()
            self.stated = True

    def advance(self, sim):
        pass

class Simulator(Graph):
    NODE_CLS = SNode
    DEFAULT_STEPPER = 'TopologicalStepper'

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)
        self.bus = {}
        self.newbus = {}
        self.running = False
        self.producers = []
        self.consumers = []
        self.stepper = StepperRegistry.ALL[self.DEFAULT_STEPPER]
        self.baserate = 30.0
        self.monitor = Monitor()

    def reset(self):
        self.tick = 0
        self.starttime = time.time()
        for pro in self.producers:
            pro.reset(self)
        for con in self.consumers:
            con.reset(self)
        self.stepper.reset(self)
    
    def advance(self):
        self.bus.update(self.newbus)
        self.newbus.clear()
        self.stepper.advance(self)

    def step(self):
        self.stepper.step(self)

    def run(self, rate=None, steps=None):
        if rate is None:
            rate = self.baserate
        if rate > 0:
            per = 1.0 / rate
        else:
            per = 0
        self.running = True
        self.reset()
        transducers = self.producers + self.consumers
        while self.running:
            stepcounts = {tran: tran.stepsNeeded(self) for tran in transducers}
            counts = max(stepcounts.values(), default=1)
            start = time.perf_counter()
            for substep in range(counts):
                self.step()
                self.advance()
                for tran in stepcounts:
                    if stepcounts[tran] > 0:
                        tran.step(self)
                self.tick += 1
            stop = time.perf_counter()
            if steps is not None and self.tick >= steps:
                return
            for tran in transducers:
                tran.advance(self)
            delay = per
            for tran in transducers:
                d = tran.delayNeeded(self)
                if d is not None:
                    delay = min(delay, d)
            if stop - start < delay:
                time.sleep(delay - (stop - start))

    def getBus(self, name):
        return self.bus.get(name)

    def setBus(self, name, val):
        self.newbus[name] = val

    def save(self):
        tree = Graph.save(self)
        root = tree.getroot()
        xconsumers = ET.SubElement(root, 'consumers')
        for con in self.consumers:
            xconsumer = ET.SubElement(xconsumers, 'consumer', name=con.name)
            for pname, val in con.props.items():
                xprop = ET.SubElement(xconsumer, 'prop', name=pname, val=str(val))
        xproducers = ET.SubElement(root, 'producers')
        for pro in self.producers:
            xproducer = ET.SubElement(xproducers, 'producer', name=pro.name)
            for pname, val in pro.props.items():
                xprop = ET.SubElement(xproducer, 'prop', name=pname, val=str(val))
        xstepper = ET.SubElement(root, 'stepper', name=type(self.stepper).__name__)
        xrate = ET.SubElement(root, 'rate', rate=str(self.baserate))
        return tree

    def restore(self, tree, schemamap, conmap, promap):
        Graph.restore(self, tree, schemamap)
        root = tree.getroot()
        xconsumers = root.find('consumers')
        if xconsumers is not None:
            for xconsumer in xconsumers.findall('consumer'):
                con = type(conmap[xconsumer.get('name')])()
                for xprop in xconsumer.findall('prop'):
                    prop = con.propmap[xprop.get('name')]
                    con.props[prop.name] = prop.type(xprop.get('val'))
                self.consumers.append(con)
        xproducers = root.find('producers')
        if xproducers is not None:
            for xproducer in xproducers.findall('producer'):
                pro = type(promap[xproducer.get('name')])()
                for xprop in xproducer.findall('prop'):
                    prop = pro.propmap[xprop.get('name')]
                    pro.props[prop.name] = prop.type(xprop.get('val'))
                self.producers.append(pro)
        xstepper = root.find('stepper')
        if xstepper is not None:
            self.stepper = StepperRegistry.ALL[xstepper.get('name')]
        xrate = root.find('rate')
        if xrate is not None:
            self.baserate = float(xrate.get('rate'))

if __name__ == '__main__':
    from lib import SchemaRegistry
    from procon import ProducerRegistry, ConsumerRegistry
    tree = ET.parse(sys.argv[1])
    sim = Simulator()
    sim.restore(tree, SchemaRegistry.ALL, ConsumerRegistry.ALL, ProducerRegistry.ALL)
    sim.run()
