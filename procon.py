import struct, time

from sim import *

class ProducerRegistry(type):
    ALL = {}
    def __new__(mcs, name, bases, dict):
        tp = type.__new__(mcs, name, bases, dict)
        if name != 'Producer':
            inst = tp()
            mcs.ALL[inst.name] = inst
        return tp

class Producer(BaseProducer, metaclass=ProducerRegistry):
    def __init__(self, name):
        self.name = name
        self.props = {}
        for prop in self.PROPS:
            if prop.default is not None:
                self.props[prop.name] = prop.default
        self.propmap = {prop.name: prop for prop in self.PROPS}

class ConsumerRegistry(type):
    ALL = {}
    def __new__(mcs, name, bases, dict):
        tp = type.__new__(mcs, name, bases, dict)
        if name != 'Consumer':
            inst = tp()
            mcs.ALL[inst.name] = inst
        return tp

class Consumer(BaseConsumer, metaclass=ConsumerRegistry):
    def __init__(self, name):
        self.name = name
        self.props = {}
        for prop in self.PROPS:
            if prop.default is not None:
                self.props[prop.name] = prop.default
        self.propmap = {prop.name: prop for prop in self.PROPS}

try:
    import rtmidi2
except ImportError:
    pass
else:
    from rtmidi2 import NOTEON, NOTEOFF, CC, PROGCHANGE, PITCHWHEEL
    class RtMIDIProducer(Producer):
        PROPS = [
            Property('notebus', str, 'midinotesi'),
            Property('voicebus', str, 'midivoices'),
            Property('progbus', str, 'midiprogsi'),
            Property('ctrlbus', str, 'midictrli'),
            Property('vpname', str, 'simula'),
            Property('client', str, ''),
        ]

        def __init__(self):
            Producer.__init__(self, 'RtMIDIProducer')

        def reset(self, sim):
            self.notes = [[0]*128 for i in range(16)]
            self.programs = [0] * 16
            self.controls = [[0]*128 for i in range(16)]
            self.voices = []
            self.midi = rtmidi2.MidiIn()
            self.midi.callback = self.message
            self.midi.open_virtual_port(self.props['vpname'])

        def message(self, msg, time):
            msgtype, channel = rtmidi2.splitchannel(msg[0])
            if msgtype == NOTEON and msg[2] == 0:
                msgtype = NOTEOFF
            if msgtype == NOTEON:
                #print('ON', msg)
                self.notes[channel][msg[1]] = msg[2]
                voice = (msg[1], msg[1], msg[2], self.programs[channel])
                if voice not in self.voices:
                    self.voices.append(voice)
            elif msgtype == NOTEOFF:
                #print('OFF', msg)
                self.notes[channel][msg[1]] = 0
                idx = 0
                while idx < len(self.voices):
                    if self.voices[idx][1] == msg[1]:
                        del self.voices[idx]
                    else:
                        idx += 1
            elif msgtype == CC:
                self.controls[channel][msg[1]] = msg[2]
            elif msgtype == PROGCHANGE:
                self.programs[channel] = (msg[1] << 7) | msg[2]

        def step(self, sim):
            sim.bus[self.props['notebus']] = self.notes
            sim.bus[self.props['progbus']] = self.programs
            sim.bus[self.props['ctrlbus']] = self.controls
            sim.bus[self.props['voicebus']] = list(map(list, self.voices))

    class RtMIDIConsumer(Consumer):
        PROPS = [
            Property('notebus', str, 'midinoteso'),
            Property('progbus', str, 'midiprogso'),
            Property('ctrlbus', str, 'midictrlo'),
            Property('vpname', str, 'simula'),
            Property('client', str, ''),
        ]

        def __init__(self):
            Consumer.__init__(self, 'RtMIDIConsumer')

        def reset(self, sim):
            self.notes = [[0]*128 for i in range(16)]
            self.programs = [0] * 16
            self.controls = [[0]*128 for i in range(16)]
            self.midi = rtmidi2.MidiOut()
            self.midi.open_virtual_port(self.props['vpname'])

        def step(self, sim):
            pass

        def advance(self, sim):
            notes = sim.bus.get(self.props['notebus'], [])
            for chan, nlists in enumerate(zip(self.notes, notes)):
                curl, nextl = nlists
                for note, pair in enumerate(zip(curl, nextl)):
                    curv, nextv = pair
                    if curv != nextv:
                        if nextv == 0:
                            self.midi.send_noteoff(chan, note)
                        else:
                            self.midi.send_noteon(chan, note, nextv)
            self.notes = notes
            controls = sim.bus.get(self.props['ctrlbus'], [])
            for chan, clists in enumerate(zip(self.controls, controls)):
                curl, nextl = clists
                for ctrl, pair in enumerate(zip(curl, nextl)):
                    curv, nextv = pair
                    if curv != nextv:
                        self.midi.send_cc(chan, ctrl, nextv)
            self.controls = controls
            programs = sim.bus.get(self.props['progbus'], [])
            for chan, progs in enumerate(zip(self.programs, programs)):
                curp, nextp = progs
                if curp != nextp:
                    self.midi.send_messages(PROGCHANGE, [chan, (nextp >> 7) & 0x7F, nextp & 0x7F])
            self.programs = programs

try:
    import pyaudio
except ImportError:
    pass
else:
    class PyAudioConsumer(Consumer):
        PROPS = [
            Property('sampbus', str, 'pasampo'),
            Property('atimebus', str, 'patimeo'),
            Property('rate', int, 44100),
            Property('channels', int, 1, min=1),
            Property('bufframes', int, 1024, min=1),
            Property('device', int, -1),
            Property('subframes', int, 0, min=0),
            Property('leadratio', float, 0.5, min=0.0, max=1.0),
        ]

        def __init__(self):
            Consumer.__init__(self, 'PyAudioConsumer')

        def reset(self, sim):
            if not hasattr(sim, 'pyaudio'):
                sim.pyaudio = pyaudio.PyAudio()
            self.stream = sim.pyaudio.open(
                rate = self.props['rate'],
                channels = self.props['channels'],
                frames_per_buffer = self.props['bufframes'],
                output_device_index = (self.props['device'] if self.props['device'] >= 0 else None),
                format = pyaudio.paFloat32,
                output = True
            )
            self.atime = 0.0
            self.buffer = []

        def step(self, sim):
            if sim.tick % (self.props['subframes'] + 1) == 0:
                try:
                    self.buffer.append(float(sim.bus.get(self.props['sampbus'], 0.0)))
                except (TypeError, ValueError):
                    self.buffer.append(0.0)
                self.atime += 1.0 / self.props['rate']
                sim.bus[self.props['atimebus']] = self.atime

        def advance(self, sim):
            samps = struct.pack('{}f'.format(len(self.buffer)), *self.buffer)
            self.stream.write(samps)
            #print('Wrote', len(self.buffer), 'samples')
            #if self.buffer:
            #    print('  avg', sum(self.buffer) / len(self.buffer))
            del self.buffer[:]
            self.prevwrite = getattr(self, 'lastwrite', time.perf_counter())
            self.lastwrite = time.perf_counter()

        def delayNeeded(self, sim):
            delay = self.props['leadratio'] * self.props['bufframes'] / self.props['rate'] - (time.perf_counter() - self.lastwrite) - (self.lastwrite - self.prevwrite)
            return delay

        def stepsNeeded(self, sim):
            #print('Need', (self.props['subframes'] + 1) * self.stream.get_write_available(), 'samples')
            return (self.props['subframes'] + 1) * self.stream.get_write_available()

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        class NumPyAudioConsumer(Consumer):
            PROPS = [
                Property('sampbus', str, 'numpasampo'),
                Property('atimebus', str, 'numpatimeo'),
                Property('rate', int, 44100),
                Property('channels', int, 1, min=1),
                Property('bufframes', int, 1024, min=1),
                Property('device', int, -1),
                Property('leadratio', float, 0.5, min=0.0, max=1.0),
            ]

            def __init__(self):
                Consumer.__init__(self, 'NumPyAudioConsumer')

            def reset(self, sim):
                if not hasattr(sim, 'pyaudio'):
                    sim.pyaudio = pyaudio.PyAudio()
                self.stream = sim.pyaudio.open(
                    rate = self.props['rate'],
                    channels = self.props['channels'],
                    frames_per_buffer = self.props['bufframes'],
                    output_device_index = (self.props['device'] if self.props['device'] >= 0 else None),
                    format = pyaudio.paFloat32,
                    output = True
                )
                self.atime = 0.0
                self.buffer = []

            def step(self, sim):
                period = 1.0 / self.props['rate']
                avail = self.stream.get_write_available()
                times = np.linspace(self.atime, self.atime + avail * period, avail, endpoint = False, dtype = np.float32)
                sim.bus[self.props['atimebus']] = times
                if len(times):
                    self.atime = times[-1] + period

            def advance(self, sim):
                frames = sim.bus[self.props['sampbus']]
                if isinstance(frames, np.ndarray) and len(frames):
                    frames = frames.astype(np.float32, copy=False)
                    self.stream.write(frames.tobytes())

            def delayNeeded(self, sim):
                if self.stream.get_write_available():
                    return 0
                return self.props['leadratio'] * self.props['bufframes'] / self.props['rate']
