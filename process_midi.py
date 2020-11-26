import pretty_midi
import numpy as np

# Function 1: midi object --> note sequence (create note sequence)
# Function 2: note sequence --> midi file (create midi file)
# Function 3: note sequence --> note sequence (eliminate double track)
# Function 4: note sequence --> note sequence (merge sequence)
# Function 5: note sequence --> quantization (from Performance RNN paper)
# Function 6: quantization --> note sequence (reverse quantization)


def create_seq(midi_object, ins_num):

    midi_list = []

    for note in midi_object.instruments[ins_num].notes:
        start = note.start
        end = note.end
        pitch = note.pitch
        velocity = note.velocity
        midi_list.append([start, end, pitch, velocity, midi_object.instruments[ins_num].program])

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    return midi_list


def create_file(seq, file_name):

    data = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(seq[0][-1])

    for event in range(len(seq)):
        note = pretty_midi.Note(start=seq[event][0], end=seq[event][1], pitch=seq[event][2], velocity=seq[event][3])
        inst.notes.append(note)

    data.instruments.append(inst)
    data.write(file_name)

    return '\nFile created!'


def elim_double(seq):

    time_list = []
    out = []

    for event in range(len(seq)):
        cur_time = seq[event][0]

        if cur_time in time_list:
            continue

        else:
            out.append(seq[event])
            time_list.append(cur_time)

    return out


def merge(seq1, seq2):
    out = seq1
    for item in seq2:
        out.append(item)
    out = sorted(out, key=lambda x: (x[0], x[2]))
    return out


def quantization(seq):
    # didn't know abt padding ... need to re-do

    quanta_list = []
    seq = np.asarray(seq).astype(type('float', (float,), {}))
    """
    --> might have to add a set_program event here in the future
    • 128 NOTE-ON events: one for each of the 128 MIDI pitches. Each one starts a new note. (0 ~ 127)
    • 128 NOTE-OFF events: one for each of the 128 MIDI pitches. Each one releases a note. (128 ~ 255)
    • 125 TIME-SHIFT events: each one moves the time step forward by increments of 8 ms up to 1 second. (256 ~ 380)
    • 128 VELOCITY events: each one changes the velocity applied to all subsequent notes (until the next velocity event).
    
    Order:
        1. Time_shift
        2. Set_veloctiy
        3. Note_on (+ pitch)
        4. Time_shift
        5. Note_off (+ pitch)
    """
    for event in range(len(seq)):

        # first time_shift value (time till note is played)
        if event == 0:
            previous_time_off = 0
        else:
            previous_time_off = seq[event-1][1]

        time_start = seq[event][0]
        wait_secs = time_start - previous_time_off
        if wait_secs >= 0:
            wait_secs /= 0.008
        else:
            wait_secs = 0

        if wait_secs > 125:
            for i in range(int(wait_secs // 125)):
                quanta_list.append(124 + 256)
            quanta_list.append(round(wait_secs % 125) + 256)
        else:
            quanta_list.append(int(wait_secs) + 256)

        # set velocity
        velocity = seq[event][3]
        quanta_list.append(int(velocity + 381))

        # note_on
        note_on = seq[event][2]
        quanta_list.append(int(note_on))

        # second time_shift (wait while note plays)
        time_off = seq[event][1]
        wait_secs = time_off - time_start
        if wait_secs >= 0:
            wait_secs /= 0.008
        else:
            wait_secs = 0

        if wait_secs >= 125:
            for i in range(int(wait_secs // 125)):
                quanta_list.append(124 + 256)
            quanta_list.append(int(wait_secs % 125) + 256)
        else:
            quanta_list.append(round(wait_secs) + 256)

        # note_off
        note_off = seq[event][2] + 128
        quanta_list.append(int(note_off))

    return quanta_list


def reverse_quanta(quanta_list, program):  # Problematic; More notes = more delay
    seq = []
    note_on, note_off, pitch_set, velocity_set = 0, 0, 0, 0
    pitch, velocity = 0, 0
    current_time = 0
    note = []

    for event in range(len(quanta_list)):

        # create new note storage
        if note_on == 1 and note_off == 1 and velocity_set == 1:
            note.append(pitch)
            note.append(velocity)
            note.append(program)
            seq.append(note)
            note = []
            note_on, note_off, velocity_set = 0, 0, 0

        # record current_time
        if 256 <= quanta_list[event] <= 380:
            current_time += (quanta_list[event] - 256) * 0.008

        # note_on
        elif quanta_list[event] <= 127 and note_on == 0:
            pitch = quanta_list[event]
            note_on = 1
            note.append(current_time)

        # note_off
        elif 128 <= quanta_list[event] <= 255 and note_on == 1:
            note_off = 1
            note.append(current_time)

        # velocity_set
        elif 381 <= quanta_list[event]:
            velocity_set = 1
            velocity = quanta_list[event] - 381

    note.append(pitch)
    note.append(velocity)
    note.append(program)
    seq.append(note)

    return seq
