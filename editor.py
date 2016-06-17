# Simula Qt node editor

import sys, traceback, os, textwrap, re
from collections import OrderedDict
import xml.etree.ElementTree as ET

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from model import *
from sim import *
from lib import SchemaRegistry
from procon import ConsumerRegistry, ProducerRegistry, Producer, Consumer

class SGraph(QGraphicsScene, Simulator):
    def __init__(self, *nodes):
        QGraphicsScene.__init__(self)
        Simulator.__init__(self)
        self.ghost = None
        for node in self.nodes:
            self.add(node)

    def add(self, node):
        Simulator.add(self, node)
        node.gnode = SGraphicNodeItem(node)
        self.addItem(node.gnode)

    def discard(self, node):
        if isinstance(node, SGraphicNodeItem):
            node = node.node
        connodes = node.connected()
        connodes.discard(node)
        Simulator.discard(self, node)
        try:
            self.removeItem(node.gnode)
        except RuntimeError:
            sys.stderr.write('Suppressed RuntimeError during node deletion; is Graph reloading?\n')
        del node.gnode
        self.updateAllWires(connodes)

    def restore(self, tree, schemamap, conmap, promap):
        Simulator.restore(self, tree, schemamap, conmap, promap)
        for node in self.nodes:
            if hasattr(node, 'co'):
                node.gnode.setPos(*node.co)
        self.updateAllWires()

    def updateAllWires(self, nodes=None):
        if nodes is None:
            nodes = self.nodes
        for node in nodes:
            for pin in node.gnode.inpins.values():
                pin.wire.updatePath()

    def embeddedSchema(self, md):
        if not md.hasFormat('application/x-qabstractitemmodeldatalist'):
            return None
        ba = md.data('application/x-qabstractitemmodeldatalist')
        ds = QDataStream(ba)
        row = ds.readUInt32()
        col = ds.readUInt32()
        items = ds.readUInt32()
        itemdata = {}
        for i in range(items):
            key, val = ds.readUInt32(), ds.readQVariant()
            itemdata[key] = val
        if Qt.UserRole not in itemdata:
            return None
        return SchemaRegistry.ALL[itemdata[Qt.UserRole]]

    def dragEnterEvent(self, ev):
        schema = self.embeddedSchema(ev.mimeData())
        node = SNode(self, schema)
        self.add(node)
        self.ghost = node.gnode
        self.ghost.setOpacity(0.5)
        self.ghost.setPos(ev.scenePos())
        ev.acceptProposedAction()

    def dragLeaveEvent(self, ev):
        if self.ghost:
            self.discard(self.ghost.node)

    def dragMoveEvent(self, ev):
        if self.ghost:
            self.ghost.setPos(ev.scenePos())
        ev.acceptProposedAction()

    def dropEvent(self, ev):
        if self.ghost:
            self.ghost.setOpacity(1)
            self.ghost = None
        ev.acceptProposedAction()

class SGraphicNodeItem(QGraphicsItem):
    WIDTH = 100
    PIN_SPC = 25
    MIN_HEIGHT = 100
    PEN_WIDTH = 1
    PEN_COLOR = QColor(0, 0, 0)
    BRUSH_COLOR = QColor(192, 192, 255)

    def __init__(self, node):
        QGraphicsItem.__init__(self)
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemIsSelectable)
        self.setFlag(self.ItemSendsScenePositionChanges)
        self.node = node
        self.inpins = OrderedDict()
        self.outpins = OrderedDict()
        self.updatePins()

    def updatePins(self):
        for pin in self.inpins.values():
            pin.setParentItem(None)
            self.scene().removeItem(pin)
        for pin in self.outpins.values():
            pin.setParentItem(None)
            self.scene().removeItem(pin)
        self.inpins.clear()
        self.outpins.clear()
        if self.node.schema:
            y = self.PIN_SPC
            for incon in self.node.schema.getIns(self.node):
                inpin = SGraphicInPinItem(self, incon)
                inpin.setParentItem(self)
                inpin.setPos(0, y)
                self.inpins[incon.name] = inpin
                y += self.PIN_SPC
            y = self.PIN_SPC
            for outcon in self.node.schema.getOuts(self.node):
                outpin = SGraphicOutPinItem(self, outcon)
                outpin.setParentItem(self)
                outpin.setPos(self.WIDTH, y)
                self.outpins[outcon.name] = outpin
                y += self.PIN_SPC
        self.height = max(self.MIN_HEIGHT, self.PIN_SPC * (1 + len(self.inpins)), self.PIN_SPC * (1 + len(self.outpins)))

    def boundingRect(self):
        ex = self.PEN_WIDTH / 2.0
        return QRectF(0, 0, self.WIDTH, self.height).adjusted(-ex, -ex, ex, ex)

    def paint(self, painter, option, widget):
        pen = QPen(self.PEN_COLOR)
        pen.setWidth(self.PEN_WIDTH)
        painter.setPen(pen)
        painter.setBrush(self.BRUSH_COLOR)
        painter.drawRect(QRectF(0, 0, self.WIDTH, self.height))
        if self.node.schema:
            fm = painter.fontMetrics()
            painter.drawText(QPointF(self.WIDTH / 2 - fm.width(self.node.schema.name) / 2, fm.ascent()), self.node.schema.name)
            if hasattr(self.node, 'props') and self.node.props:
                y = self.height - fm.descent()
                for name, val in sorted(self.node.props.items(), key=lambda pair: pair[0]):
                    t = '{}: {}'.format(name, val)
                    painter.drawText(QPointF(self.WIDTH / 2 - fm.width(t) / 2, y), t)
                    y -= fm.height()
                if y < fm.height():
                    self.height += (fm.height() - y)

    def itemChange(self, change, val):
        if change == self.ItemScenePositionHasChanged:
            for pin in self.inpins.values():
                pin.wire.updatePath()
            for pin in self.outpins.values():
                for to, in_ in self.node.outgoing[pin.con.name]:
                    try:
                        to.gnode.inpins[in_].wire.updatePath()
                    except KeyError:
                        print('Suppressed KeyError routing {} on {} of {}; is the graph reloading?'.format(in_, to.schema, to))
                        to.gnode.updatePins()
        return val

    def mouseDoubleClickEvent(self, ev):
        proped = SNodePropertyEditor(self.scene().views()[0], self.node)
        proped.show()

class SGraphicPinItem(QGraphicsEllipseItem):
    PIN_RAD = 10
    PIN_COLOR = QColor(0, 0, 0)
    TYPE_COLORS = {
        int: QColor(0, 0, 255),
        float: QColor(127, 0, 255),
        str: QColor(0, 255, 0),
        bytes: QColor(0, 127, 0),
        list: QColor(255, 255, 0),
        dict: QColor(255, 127, 0),
        bool: QColor(255, 0, 0),
    }
    PIN_POLY_COLOR = QColor(0, 255, 255)
    GHOST_COLOR = QColor(255, 0, 255)
    GHOST_Z = 2
    LABEL_Z = 3

    def __init__(self, gnode, con):
        pr = self.PIN_RAD
        QGraphicsEllipseItem.__init__(self, QRectF(-pr, -pr, 2*pr, 2*pr))
        self.setFlag(self.ItemIsSelectable)
        if con.unique:
            self.setPen(QPen(Qt.NoPen))
        else:
            self.setPen(self.PIN_POLY_COLOR)
        col = self.TYPE_COLORS.get(con.type, self.PIN_COLOR)
        invcol = QColor(255-col.red(), 255-col.green(), 255-col.blue())
        self.setBrush(col)
        self.gnode = gnode
        self.con = con
        self.label = QGraphicsSimpleTextItem(self.con.name, self)
        self.label.setBrush(invcol)
        fm = QFontMetrics(self.label.font())
        self.label.setPos(-fm.width(self.con.name) / 2, -fm.height() / 2)
        self.label.setZValue(self.LABEL_Z)

    def mousePressEvent(self, ev):
        ev.accept()

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            if not hasattr(self, 'ghost'):
                self.ghost = QGraphicsPathItem(self)
                self.ghost.setPen(self.GHOST_COLOR)
                self.ghost.setZValue(self.GHOST_Z)
            path = QPainterPath(QPointF(0, 0))
            path.lineTo(ev.pos())
            self.ghost.setPath(path)

    def mouseReleaseEvent(self, ev):
        if hasattr(self, 'ghost'):
            self.ghost.setParentItem(None)
            del self.ghost
        graph = self.scene()
        others = graph.items(ev.scenePos())
        for other in others:
            if isinstance(other, SGraphicPinItem):
                out = None
                in_ = None
                if isinstance(self, SGraphicOutPinItem):
                    out = self
                if isinstance(other, SGraphicOutPinItem):
                    out = other
                if isinstance(self, SGraphicInPinItem):
                    in_ = self
                if isinstance(other, SGraphicInPinItem):
                    in_ = other
                if not (out and in_):
                    return
                if out.gnode.node.connected(out.con.name, in_.gnode.node, in_.con.name):
                    out.gnode.node.disconnect(out.con.name, in_.gnode.node, in_.con.name)
                else:
                    out.gnode.node.connect(out.con.name, in_.gnode.node, in_.con.name)
                in_.wire.updatePath()
                break

try:
    import numpy
except ImportError:
    pass
else:
    SGraphicPinItem.TYPE_COLORS[numpy.ndarray] = QColor(127, 127, 0)

class SGraphicWireItem(QGraphicsPathItem):
    WIRE_WIDTH = 2
    WIRE_COLOR = QColor(0, 0, 127)
    WIRE_Z = 1

    def __init__(self, pin):
        QGraphicsPathItem.__init__(self)
        self.setFlag(self.ItemIsSelectable)
        self.setZValue(self.WIRE_Z)
        pen = QPen(self.WIRE_COLOR)
        pen.setWidth(self.WIRE_WIDTH)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.pin = pin
        self.updatePath()

    def updatePath(self):
        path = QPainterPath()
        for pt in self.pin.connectedPoints():
            pt = pt.center()
            path.moveTo(pt)
            if abs(pt.x()) < 5:
                path.lineTo(0, 0)
            else:
                path.cubicTo(pt.x() / 2, pt.y(), pt.x() / 2, 0, 0, 0)
        try:
            self.setPath(path)
        except RuntimeError:
            sys.stderr.write('Suppressed RuntimeError during wire pathing; is Graph reloading?\n')

class SGraphicInPinItem(SGraphicPinItem):
    def __init__(self, gnode, con):
        SGraphicPinItem.__init__(self, gnode, con)
        self.wire = SGraphicWireItem(self)
        self.wire.setParentItem(self)

    def connectedPoints(self):
        for node, out in self.gnode.node.incoming[self.con.name]:
            try:
                yield self.mapRectFromItem(node.gnode.outpins[out], QRectF(-1, -1, 2, 2))
            except RuntimeError:
                sys.stderr.write('Suppressed RuntimeError during connection mapping; is Graph reloading?\n')
            except KeyError:
                sys.stderr.write('Suppressed KeyError of {} on {} of {} during connection mapping; is Graph reloading?\n'.format(out, node.schema, node))
                node.gnode.updatePins()

class SGraphicOutPinItem(SGraphicPinItem):
    def __init__(self, gnode, con):
        SGraphicPinItem.__init__(self, gnode, con)

class SGraphicView(QGraphicsView):
    def __init__(self, graph, schemata, fname=None):
        QGraphicsView.__init__(self, graph)
        self.graph = graph
        self.schemata = schemata
        self.fname = fname
        self.ctxmenu = QMenu()
        self.nodemenu = QMenu()
        self.updateContextMenu()
        self.setResizeAnchor(self.AnchorUnderMouse)
        self.setDragMode(self.ScrollHandDrag)
        self.setAcceptDrops(True)

    def updateContextMenu(self):
        self.ctxmenu.clear()
        self.schmenu = self.ctxmenu.addMenu('Add')
        self.catmenus = {}
        for schema in sorted(self.schemata, key=lambda x: (getattr(x, 'cat', 'Uncategorized'), type(x).__name__)):
            cat = getattr(schema, 'cat', 'Uncategorized')
            if cat not in self.catmenus:
                self.catmenus[cat] = self.schmenu.addMenu(cat)
            self.catmenus[cat].addAction(type(schema).__name__, lambda self=self, schema=schema: self.createNew(schema))
        self.nodemenu.clear()
        self.nodemenu.addAction('Properties', self.ndprops)
        self.nodemenu.addAction('Remove', self.ndremove)

    def ndprops(self):
        proped = SNodePropertyEditor(self, self.modnode)
        proped.show()

    def ndremove(self):
        self.graph.discard(self.modnode)
        self.modnode = None

    def contextMenuEvent(self, ev):
        self.addpt = self.mapToScene(ev.pos())
        node = self.graph.itemAt(self.addpt, QTransform())
        if isinstance(node, SGraphicNodeItem):
            self.modnode = node.node
            self.nodemenu.popup(ev.globalPos())
        else:
            self.ctxmenu.popup(ev.globalPos())

    def wheelEvent(self, ev):
        d = ev.pixelDelta().y()
        self.scale(0.999**d, 0.999**d)

    def createNew(self, schema, pt=None):
        if not pt:
            pt = self.addpt
        nd = SNode(self.graph, schema)
        self.graph.add(nd)
        nd.gnode.setPos(pt)
        return nd

class SEditorWindow(QMainWindow):
    FD_FILTER = 'Simulation Files (*.sim);;All Files (*)'
    TITLE = 'Simula Editor'

    def __init__(self):
        QMainWindow.__init__(self)
        self.icons = {
            'sim': QIcon('res/sim.svg'),
            'sim_dirty': QIcon('res/sim_dirty.svg'),
        }
        self.frame = QWidget()
        self.fbox = QHBoxLayout()
        self.frame.setLayout(self.fbox)
        self.palette = QTreeWidget()
        self.palette.setColumnCount(1)
        self.palette.setHeaderLabels(['Schemas'])
        self.palette.setMinimumWidth(250)
        self.palette.setMaximumWidth(250)
        self.palette.setDragEnabled(True)
        self.palcats = {}
        self.palitems = {}
        for cat in sorted(set((i.cat for i in SchemaRegistry.ALL.values()))):
            self.palcats[cat] = QTreeWidgetItem(self.palette, [cat])
        for schema in sorted(SchemaRegistry.ALL.values(), key=lambda x: x.name):
            self.palitems[schema.name] = QTreeWidgetItem(self.palcats[schema.cat], [type(schema).__name__])
            self.palitems[schema.name].setData(0, Qt.UserRole, schema.name)
            self.palitems[schema.name].setToolTip(0, self.docstring(schema))
        self.fbox.addWidget(self.palette)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.closeTab)
        self.fbox.addWidget(self.tabs)
        self.setCentralWidget(self.frame)
        self.setWindowTitle(self.TITLE)
        self.fmenu = self.menuBar().addMenu('File')
        self.fmenu.addAction('New', self.fnew)
        self.fmenu.addAction('Save', self.fsave)
        self.fmenu.addAction('Save As', self.fsaveas)
        self.fmenu.addAction('Load', self.fload)
        self.fmenu.addSeparator()
        self.fmenu.addAction('Quit', self.fquit)
        self.smenu = self.menuBar().addMenu('Simulation')
        self.smenu.addAction('Properties', self.sprop)

    def docstring(self, schema):
        doc = type(schema).__doc__
        if not doc:
            return '<i>No documentation.</i>'
        doc = doc.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        doc = textwrap.fill(doc).replace('\n', '<br/>')
        doc = re.sub("\\`([^']*)\\'", '<i>\\1</i>', doc)
        return doc

    def closeTab(self, idx):
        self.tabs.widget(idx).close()
        self.tabs.removeTab(idx)

    def addView(self, view):
        self.tabs.addTab(view, self.icons['sim'], ('unnamed' if view.fname is None else os.path.basename(view.fname)))

    def fnew(self):
        self.tabs.addTab(SGraphicView(SGraph(), SchemaRegistry.ALL.values()), self.icons['sim'], 'unnamed')

    def fsave(self):
        tab = self.tabs.currentWidget()
        if not tab:
            sys.stderr.write('...OK, no tab to save. Stop that.\n')
            return
        if tab.fname is not None:
            tab.graph.save().write(tab.fname)
        else:
            self.fsaveas()

    def fsaveas(self):
        tab = self.tabs.currentWidget()
        if not tab:
            sys.stderr.write('...OK, no tab to save. Stop that.\n')
            return
        fname = QFileDialog.getSaveFileName(self, 'Save As', '', self.FD_FILTER)[0]
        tab.fname = fname
        self.tabs.setTabText(self.tabs.currentIndex(), os.path.basename(tab.fname))
        self.fsave()

    def fload(self):
        fname = QFileDialog.getOpenFileName(self, 'Open', '', self.FD_FILTER)[0]
        if not fname:
            return
        graph = SGraph()
        graph.restore(ET.parse(fname), SchemaRegistry.ALL, ConsumerRegistry.ALL, ProducerRegistry.ALL)
        self.tabs.addTab(SGraphicView(graph, SchemaRegistry.ALL.values(), fname), self.icons['sim'], os.path.basename(fname))

    def fquit(self):
        exit()

    def sprop(self):
        tab = self.tabs.currentWidget()
        if not tab:
            sys.stderr.write('...OK, no tab to take properties of. Stop that.\n')
            return
        simed = SSimConfigEditor(self, tab.graph)
        simed.show()

class SNodePropertyEditor(QDialog):
    def __init__(self, parent, node):
        QDialog.__init__(self, parent)
        self.node = node
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.setLayout(self.grid)
        self.setWindowTitle('Node Property Editor of {}'.format(self.node.schema.name))
        if not (node.schema and hasattr(node.schema, 'props') and node.schema.props):
            msg = QLabel('This node does not have modifiable properties.')
            self.grid.addWidget(msg, 0, 0)
        else:
            self.buildProps(node.schema.props)

    def buildProps(self, props):
        self.props = props
        self.propeds = {}
        y = 0
        for prop in props:
            msg = QLabel(prop.name)
            self.grid.addWidget(msg, y, 0)
            val = self.node.getProp(prop.name)
            ed = QLineEdit()
            ed.setText(str(val))
            self.grid.addWidget(ed, y, 1)
            self.propeds[prop.name] = ed
            y += 1
        app = QPushButton('Apply')
        app.clicked.connect(self.apply)
        self.grid.addWidget(app, y, 0, 1, 2)

    def apply(self):
        for prop in self.props:
            ed = self.propeds[prop.name]
            try:
                self.node.props[prop.name] = prop.type(ed.text())
            except Exception:
                print('===== Exception while setting value {} of {} of {} ====='.format(prop.name, self.node.schema, self.node))
                traceback.print_exc()
        self.node.gnode.updatePins()
        self.node.gnode.update()
        self.close()

class SSimListModel(QAbstractItemModel):
    def __init__(self, graph):
        QAbstractItemModel.__init__(self)
        self.graph = graph
        self.indices = set()
        self.indexmap = {}

    def data(self, index, role):
        if not index.isValid():
            return QVariant()
        print(self.indexmap)
        try:
            obj = self.indexmap[index.internalId()]
        except KeyError:
            obj = self.indexmap[index.internalPointer()]
        if isinstance(obj, (Consumer, Producer)):
            if role == Qt.DisplayRole:
                return type(obj).__name__
        elif isinstance(obj, tuple):
            if len(obj) == 3:
                if role == Qt.DisplayRole:
                    return [obj[2], obj[0].props[obj[2]]][index.column()]
        return QVariant()

    def headerData(self, section, orient, role):
        if orient != Qt.Horizontal or section not in (0, 1):
            return QVariant()
        if role == Qt.DisplayRole:
            return [self.LIST_NAME, 'Value'][section]
        return QVariant()

    def index(self, row, col, parent):
        print('index: parent valid', parent.isValid(), 'row', parent.row(), 'col', parent.column())
        print('  asking row', row, 'col', col)
        if not parent.isValid():
            if col != 0 or row >= len(self.graphList()):
                return QModelIndex()
            obj = self.graphList()[row]
            if obj not in self.indices:
                self.indices.add(obj)
                print('  registered object id', id(obj))
                self.indexmap[id(obj)] = obj
            return self.createIndex(row, col, id(obj))
        else:
            obj = self.indexmap[parent.internalId()]
            print('internal obj:', obj)
            if isinstance(obj, (Consumer, Producer)):
                if col not in (0, 1) or row >= len(obj.props):
                    return QModelIndex()
                subobj = (obj, parent.row(), sorted(obj.props.keys())[row])
                if subobj not in self.indices:
                    self.indices.add(subobj)
                    print('  registered object id', id(subobj))
                    self.indexmap[id(subobj)] = subobj
                return self.createIndex(row, col, id(subobj))
        return QModelIndex()

    def parent(self, index):
        print('<parent>')
        print('parent: index valid', index.isValid(), 'row', index.row(), 'col', index.column(), 'int', self.indexmap[index.internalId()])
        if not index.isValid():
            return QModelIndex()
        obj = self.indexmap[index.internalId()]
        if isinstance(obj, tuple):
            if len(obj) == 3:
                return self.createIndex(obj[1], 0, obj[0])
        return QModelIndex()

    def rowCount(self, index):
        if not index.isValid():
            return len(self.graphList())
        obj = self.indexmap[index.internalId()]
        if isinstance(obj, (Producer, Consumer)):
            return len(obj.props)
        return 0

    def columnCount(self, index):
        if not index.isValid():
            return 1
        obj = self.indexmap[index.internalId()]
        if isinstance(obj, (Producer, Consumer)):
            return 2
        return 0

    def append(self, item):
        gl = self.graphList()
        self.beginInsertRows(QModelIndex(), len(gl), len(gl))
        gl.append(item)
        self.endInsertRows()

class SSimProModel(SSimListModel):
    LIST_NAME = 'Producers'

    def graphList(self):
        return self.graph.producers

class SSimConModel(SSimListModel):
    LIST_NAME = 'Consumers'

    def graphList(self):
        return self.graph.consumers

class SSimTreeView(QTreeView):
    def __init__(self, mdl, editor):
        QTreeView.__init__(self)
        self.mdl = mdl
        self.editor = editor
        self.setModel(mdl)

    def mouseDoubleClickEvent(self, ev):
        for midx in self.selectedIndexes():
            ed = self.editor(self, self.mdl.graphList()[midx.row()])
            ed.show()

class SSimConfigEditor(QDialog):
    def __init__(self, parent, graph):
        QDialog.__init__(self, parent)
        self.graph = graph
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.setLayout(self.grid)
        self.setWindowTitle('Simulation Editor')
        self.promdl = SSimProModel(self.graph)
        self.conmdl = SSimConModel(self.graph)
        self.prolv = SSimTreeView(self.promdl, SProconPropertyEditor)
        self.conlv = SSimTreeView(self.conmdl, SProconPropertyEditor)
        self.grid.addWidget(self.prolv, 0, 0)
        self.grid.addWidget(self.conlv, 0, 1)
        self.proadd = QPushButton('Add')
        self.proadd.clicked.connect(self.padd)
        self.grid.addWidget(self.proadd, 1, 0)
        self.conadd = QPushButton('Add')
        self.conadd.clicked.connect(self.cadd)
        self.grid.addWidget(self.conadd, 1, 1)
        self.stepperlab = QLabel('Stepper:')
        self.grid.addWidget(self.stepperlab, 2, 0)
        self.stepperlw = QListWidget()
        for name in sorted(StepperRegistry.ALL.keys()):
            self.stepperlw.addItem(name)
        self.stepperlw.setCurrentItem(self.stepperlw.findItems(type(self.graph.stepper).__name__, Qt.MatchFixedString | Qt.MatchCaseSensitive)[0])
        self.stepperlw.currentItemChanged.connect(self.slwchange)
        self.grid.addWidget(self.stepperlw, 3, 0)
        self.ratelab = QLabel('Rate:')
        self.grid.addWidget(self.ratelab, 2, 1)
        self.rateed = QLineEdit()
        self.rateed.setText(str(self.graph.baserate))
        self.rateed.editingFinished.connect(self.rechange)
        self.grid.addWidget(self.rateed, 3, 1, Qt.AlignTop)
        self.pmenu = QMenu()
        self.cmenu = QMenu()
        self.buildMenus()

    def buildMenus(self):
        self.pmenu.clear()
        for pro in sorted(ProducerRegistry.ALL.values(), key=lambda x: x.name):
            self.pmenu.addAction(pro.name, lambda self=self, pro=pro: self.addProducer(pro))
        self.cmenu.clear()
        for con in sorted(ConsumerRegistry.ALL.values(), key=lambda x: x.name):
            self.cmenu.addAction(con.name, lambda self=self, con=con: self.addConsumer(con))

    def padd(self):
        self.pmenu.popup(QCursor.pos())

    def cadd(self):
        self.cmenu.popup(QCursor.pos())

    def addProducer(self, pro):
        self.promdl.append(type(pro)())

    def addConsumer(self, con):
        self.conmdl.append(type(con)())

    def slwchange(self, current, old):
        self.graph.stepper = StepperRegistry.ALL[current.text()]

    def rechange(self):
        self.graph.baserate = float(self.rateed.text())

class SProconPropertyEditor(QDialog):
    def __init__(self, parent, procon):
        QDialog.__init__(self, parent)
        self.procon = procon
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.setLayout(self.grid)
        self.setWindowTitle('Producer/Consumer Property Editor of {}'.format(type(procon).__name__))
        self.buildProps(procon.PROPS)

    def buildProps(self, props):
        self.props = props
        self.propeds = {}
        y = 0
        for prop in props:
            msg = QLabel(prop.name)
            self.grid.addWidget(msg, y, 0)
            val = self.procon.props[prop.name]
            ed = QLineEdit()
            ed.setText(str(val))
            self.grid.addWidget(ed, y, 1)
            self.propeds[prop.name] = ed
            y += 1
        app = QPushButton('Apply')
        app.clicked.connect(self.apply)
        self.grid.addWidget(app, y, 0, 1, 2)

    def apply(self):
        for prop in self.props:
            ed = self.propeds[prop.name]
            try:
                self.procon.props[prop.name] = prop.type(ed.text())
            except Exception:
                print('===== Exception while setting value {} of {} of {} ====='.format(prop.name, type(self.procon).__name__, self.procon))
                traceback.print_exc()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('res/simula.svg'))
    graph = SGraph()
    win = SEditorWindow()
    if len(sys.argv) < 2:
        sc = SchemaRegistry.ALL['ADD']
        nd1 = SNode(graph, sc)
        nd2 = SNode(graph, sc)
        nd1.connect('Q', nd2, 'I')
        graph.add(nd1)
        graph.add(nd2)
        nd2.gnode.setPos(QPointF(150, 50))
        view = SGraphicView(graph, SchemaRegistry.ALL.values())
        win.addView(view)
    else:
        tree = ET.parse(sys.argv[1])
        graph.restore(tree, SchemaRegistry.ALL, ConsumerRegistry.ALL, ProducerRegistry.ALL)
        view = SGraphicView(graph, SchemaRegistry.ALL.values(), sys.argv[1])
        win.addView(view)
    win.show()
    exit(app.exec_())
