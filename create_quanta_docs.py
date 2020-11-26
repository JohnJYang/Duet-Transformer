from process_midi import create_seq, quantization
import pretty_midi
import glob

# Function 1: quanta_list, string --> quanta_doc (write quanta_doc from list)
# Function 2: midi files (processed) --> quanta_list_1 and _2 (create quanta_list from midi files)


def write_quanta_doc(input, name):
    doc = open(name + '.txt', 'w')
    for line in input:
        for event in line:
            doc.write(str(event) + ',')
        doc.write('\n')
    doc.close()


def create_quanta_list(folder_path):
    quanta_list_0 = []
    quanta_list_1 = []
    for i in range(2):
        for file in sorted(glob.glob(folder_path + "/*_" + str(i) + ".mid"), key=lambda x: int(x[len(folder_path) + 1:-6])):
            midi_data = pretty_midi.PrettyMIDI(file)
            quanta = quantization(create_seq(midi_data, 0))
            if i == 0:
                quanta_list_0.append(quanta)
            elif i == 1:
                quanta_list_1.append(quanta)
    return quanta_list_0, quanta_list_1


'''
# --> Commands for creating quanta_docs
write_quanta_doc(create_quanta_list("data/separated/alto_saxs")[0], 'alto_saxs_0')
write_quanta_doc(create_quanta_list("data/separated/alto_saxs")[1], 'alto_saxs_1')
write_quanta_doc(create_quanta_list("data/separated/cellos")[0], 'cellos_0')
write_quanta_doc(create_quanta_list("data/separated/cellos")[1], 'cellos_1')
write_quanta_doc(create_quanta_list("data/separated/clarinets")[0], 'clarinets_0')
write_quanta_doc(create_quanta_list("data/separated/clarinets")[1], 'clarinets_1')
write_quanta_doc(create_quanta_list("data/separated/flutes")[0], 'flutes_0')
write_quanta_doc(create_quanta_list("data/separated/flutes")[1], 'flutes_1')
write_quanta_doc(create_quanta_list("data/separated/french_horns")[0], 'french_horns_0')
write_quanta_doc(create_quanta_list("data/separated/french_horns")[1], 'french_horns_1')
write_quanta_doc(create_quanta_list("data/separated/trombones")[0], 'trombones_0')
write_quanta_doc(create_quanta_list("data/separated/trombones")[1], 'trombones_1')
write_quanta_doc(create_quanta_list("data/separated/trumpets")[0], 'trumpets_0')
write_quanta_doc(create_quanta_list("data/separated/trumpets")[1], 'trumpets_1')
write_quanta_doc(create_quanta_list("data/separated/tubas")[0], 'tubas_0')
write_quanta_doc(create_quanta_list("data/separated/tubas")[1], 'tubas_1')
write_quanta_doc(create_quanta_list("data/separated/violins")[0], 'violins_0')
write_quanta_doc(create_quanta_list("data/separated/violins")[1], 'violins_1')
'''