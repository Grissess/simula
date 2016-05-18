# Simula node model

import sys

from collections import defaultdict
import xml.etree.ElementTree as ET

class Node(object):
    def __init__(self, graph=None, schema=None):
        self.graph = graph
        self.schema = schema
        self.outgoing = defaultdict(set)
        self.incoming = defaultdict(set)

    def connect(self, out, to, in_):
        if to.schema:
            sin = to.schema.getInMap(to).get(in_)
            if sin and sin.unique:
                for source, out in to.incoming[in_].copy():
                    source.disconnect(out, to, in_)
        self.outgoing[out].add((to, in_))
        to.incoming[in_].add((self, out))

    def connected(self, out=None, to=None, in_=None):
        if out is None:
            ret = set()
            for name, edges in self.outgoing.items():
                for to, in_ in edges:
                    ret.add(to)
            for name, edges in self.incoming.items():
                for from_, out in edges:
                    ret.add(from_)
            return ret
        edges = self.outgoing[out]
        if not to:
            return edges
        edges = set([edge for edge in edges if edge[0] == to])
        if not in_:
            return edges
        return (to, in_) in edges

    def disconnect(self, out, to, in_):
        self.outgoing[out].discard((to, in_))
        to.incoming[in_].discard((self, out))

    def expunge(self):
        for name, edges in list(self.outgoing.items()):
            for to, in_ in edges.copy():
                self.disconnect(name, to, in_)
        for name, edges in list(self.incoming.items()):
            for from_, out in edges.copy():
                from_.disconnect(out, self, name)

    def save(self, root):
        this = ET.SubElement(root, 'node', schema=self.schema.name, id=str(id(self)))
        if hasattr(self, 'gnode'):
            try:
                xco = ET.SubElement(this, 'co', x=str(self.gnode.pos().x()), y=str(self.gnode.pos().y()))
            except RuntimeError:
                sys.stderr.write('Suppressed RuntimeError during coordinate save; is Graph reloading?\n')
        for name, edges in self.outgoing.items():
            xout = ET.SubElement(this, 'out', name=name)
            for node, in_ in edges:
                xedge = ET.SubElement(xout, 'edge', to=str(id(node)))
                xedge.set('in', in_)
        return this

    def restore(self, this, idmap):
        xco = this.find('co')
        if xco is not None:
            self.co = (float(xco.get('x')), float(xco.get('y')))
        for xout in this.findall('out'):
            for xedge in xout.findall('edge'):
                self.connect(xout.get('name'), idmap[int(xedge.get('to'))], xedge.get('in'))

    def dump(self):
        print('  {} - {}:'.format(self, self.schema))
        for name, edges in self.outgoing.items():
            print('    {}'.format(name))
            for to, in_ in edges:
                print('      {} of {} - {}'.format(in_, to, to.schema))

class Graph(object):
    NODE_CLS = Node

    def __init__(self, *nodes):
        self.nodes = set(nodes)
    
    def add(self, node):
        self.nodes.add(node)
        node.graph = self

    def discard(self, node):
        node.expunge()
        self.nodes.discard(node)
        node.graph = None

    def clear(self):
        for node in self.nodes.copy():
            self.discard(node)
        self.nodes.clear()

    def dump(self):
        print('-----')
        for node in self.nodes:
            node.dump()
        print('-----')

    def save(self):
        graph = ET.Element('graph')
        tree = ET.ElementTree(graph)
        nodes = ET.SubElement(graph, 'nodes')
        for node in self.nodes:
            node.save(nodes)
        return tree

    def restore(self, tree, schemamap):
        Graph.clear(self)
        graph = tree.getroot()
        nodes = graph.find('nodes')
        idmap = {}
        for xnode in nodes.findall('node'):
            node = self.NODE_CLS(self, schemamap[xnode.get('schema')])
            idmap[int(xnode.get('id'))] = node
            self.add(node)
        for xnode in nodes.findall('node'):
            idmap[int(xnode.get('id'))].restore(xnode, idmap)

class Connector(object):
    def __init__(self, name, type=None, unique=True):
        self.name = name
        self.type = type
        self.unique = unique

    def __repr__(self):
        return '<Conn {}:{}:{}>'.format(self.name, self.type, 'unique' if self.unique else 'poly')

class Schema(object):
    def __init__(self, name, ins, outs):
        self.name = name
        self.ins = ins  # [Connector]
        self.outs = outs  # [Connector]
        self.inmap = {con.name: con for con in self.ins}
        self.outmap = {con.name: con for con in self.outs}

    def getIns(self, node):
        return self.ins

    def getInMap(self, node):
        return self.inmap

    def getOuts(self, node):
        return self.outs

    def getOutMap(self, node):
        return self.outmap

    def __repr__(self):
        return '<Schema {} in:{} out:{}>'.format(self.name, self.ins, self.outs)
