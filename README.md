# Simula

Simula is a few things:

- A network modeling library.
- A dataflow programming language based on this network library.
- A graphical editor for networks, presently designed to be an editor for said
  programming language.

## Network Model

The Simula network model is defined in `model.py`, and contains the core
classes for that model.

All of the classes are centered around the `Node`, which is a vertex in a
graph. All `Node`s have directed, labelled edges stored in members `incoming`
and `outgoing` as maps of sets. the labels come in pairs, intuitively
representing the label exiting one vertex and the label entering another; each
element of the set is a `(vertex, vertlabel)` tuple. It is an invariant that, if
`(A, x)` is in `B.outgoing[y]`, then `(B, y)` is in `A.incoming[x]`.

`Node`s are typically contained in a `Graph`, which is little more than a set
of `Node`s. `Graph`s allow a network topology to be saved to an `ElementTree`
via the `.save()` method, and later restored from an `ElementTree` using the
`.restore()` method. Both of these methods call implementations of the same in
the `Graph`'s `Node`s, which are expected to serialize the node to an XML
`Element`.  Subclasses may override these methods (and call the base
implementation first); subclasses of `Graph` may also set the `NODE_CLS` class
member to the type of `Node` or derivative that it would like to use in
`.restore()`. The `NODE_CLS` constructor should have the same parameters as
`Node`.

To assist with network design and validation, `Node`s have a `schema` member,
which, if not `None`, is a `Schema` object, which has an identifying `name` and
methods `.getIns()` and `.getOuts()` which return the expected incoming labels
and outgoing labels for edges to nodes with this schema as lists of
`Connector`s. A `Connector` is a simple immutable class which contains a `name`
(which functions as the label) and other optional parameters (as `type` and
`unique`) mostly used by the language. For convenience, `Schema`s also possess
`.getInMap()` and `.getOutMap()` methods which return mappings from names to
corresponding `Connector`s, though ordering information is lost.

## Dataflow Programming Language

The Simula executive (`Simulator`, `Stepper`, etc.) is in `sim.py`. Schemata
compatible with the language may be found in `lib.py`, which imports other
`lib_*.py` files based on the presence of certain supporting Python modules.
Many `Producer`s and `Consumer`s can be found in `procon.py`.

Executable graphs, or `Simulators`, inherit from `Graph` directly, and so can
be saved and restored as with `Graph`. The node class is `SNode`, a simulation
node, which extends `Node` by having members `values`, `newvalues`, and
`props`; the former two are used to hold current and next-tick values for
outputs from this `SNode`, and the latter contains the values assigned to this
node's properties (see below).

To execute a `Simulator`, first call `.reset()`, then `.run()`. If you would
like to only run it for a certain number of ticks (or until some condition),
instead call `.step()` followed by `.advance()` for each tick.

`Simulator`s have, in addition to `SNode`s, busses (member `bus` and `newbus`),
`Producer`s, `Consumer`s, a `Stepper`, and a `Monitor`. The busses are accessed
from within the language through the `GetBus` and `SetBus` simulation schemata,
and are typically used to communicate values between the outside environment
and the simulation.  `Producer`s and `Consumer`s (generally "transducers") are
similar to `SNode`s, but can influence the simulation rate (via
`.stepsNeeded()` and `.delayNeeded()`), and are typically used to interact with
some environment or other library. The `Stepper` determines how (and in what
order) `SNode`s' `.step()` and `.advance()` methods are called, and the
`Monitor` reports errors during the simulation in a flexible manner.

A `SNode`'s functional behavior is almost exclusively managed by its
`SimSchema`, a `Schema` derivative that has support for `Property` objects and
an additional category (`cat`) member. `Property` objects represent schemata
for properties; various objects (including `Simulator`s, `SNode`s, transducers,
etc.) have a `props` member which contains keys corresponding to a
`Property` `name`, whose type is that of the `type` member of the
`Property`, and whose value is initialized to `default` if not otherwise
set. The `Property` constructor also accepts arbitrary keyword arguments,
which become members of the object; this may be used to hint legal values
(such as `min` and `max` for numeric types) or encode documentation.

The file `sim.py` will, when called as a script, load a simulation from a file
specified as its only argument and `.run()` it until interrupted.

## Network Editor

The Simula editor is completely contained in `editor.py`, but depends on all
other modules. It is based on PyQt5 (and so requires Qt5 to run). Called as a
script, `editor.py` opens the editor window. If given a filename as an
argument, it loads that initially; otherwise, it generates a sample simulation.

The editor is a multi-document interface with tabs and a palette; most of the
interface is centered about the graphics view in the center, which visualizes
the network and can be scrolled by clicking and dragging, or zoomed using the
mousewheel. New nodes (with specified schemata) can be dragged in from the
palette to the left, or created by invoking the context menu in an empty part
of the scene and selecting "Add". Hovering over a schema in the palette will
show its documentation, derived from the schema's docstring. Invoking the
context menu on a specific node will allow one to access the "Properties"
dialog or remove the node; properties can also be accessed by double-clicking
the node. The "pins" of a node correspond to the inputs and outputs of its
schema, and are ordered top to bottom; inputs are on the left, and outputs are
on the right. Pins are colored for the type of data they produce, but the type
system is presently mostly advisory. A pin with a cyan border is a "polyinput"
that can accept multiple input connections; most other pins (without this
border) can only accept one connection. To create a connection, click and drag
between an output pin and an input pin; to delete a connection, click and drag
between the connected pins. A wire, drawn as a curve, will indicate which
connections are made.

Simulation properties may be accessed via "Properties" in the "Simulation" menu
on the menubar.  Soon, a simulation control dialog will also be accessible
therein.

The "File" menu of the menubar contains the functionality to save and load the
simulations, as well as open new, blank simulations. Files so saved can later
be run with `sim.py`, and be edited with any standard XML or text editor.
"Quit" will exit the editor.

## TODO

- Improve documentation
- Add more (useful) functionality to the programming language, including
  schemata and transducers for various Python data-handling libraries
- Libraries which need to be added or improved: socket, numpy, scipy, pyaudio,
  rtmidi2, PyQt5 (simulation-populated GUIs and controls); more may be
  requested
- Decouple the editor and programming language, such that the editor can be
  used to make "plain old" networks as needed.
- Allow the editor to run simulations
- Fix TODOs in various interface code
- Standardize schemata for transducers
- Standardize category and node namespace, especially for the `lib_*.py`
  extension libraries
- Improve handling of "null" (`None`) values in the simulation, and provide
  more intuitive input synchronization methods
- Make subsimulation creation, editing, and interaction more intuitive (display
  editor in editor? display a file select dialog for the property?)
- Make the property editors in all places more intuitive and expressive
- Detect modifications to simulations in the editor in non-obnoxious,
  easily-handled ways
- Create a standard library of common, optimized functions/simulations as
  either files or nodes
- Test and optimize :)
