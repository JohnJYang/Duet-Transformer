from process_midi import create_seq, create_file, elim_double, merge, quantization
import pretty_midi
import glob

# Function 1: log_list, string --> txt. file (log during file creation to spot any abnormal files)
# Function 2: path, instrument index, instrument index, path, number --> file (use to fix abnormal files)
# Function 3: path, path, string --> midi files + log_list (create separated midi tracks)


def write_log(log_list, name):
    file = open(name + '.txt', 'w')
    for line in log_list:
        file.write(str(line[0]) + ' ' + line[1] + ' ' + line[2])
        file.write('\n\n')
    file.close()


def fix_file(file, index_1, index_2, out_path, count):
    midi_data = pretty_midi.PrettyMIDI(file)
    create_file(elim_double(create_seq(midi_data, index_1)), out_path + str(count) + "_0.mid")
    create_file(elim_double(create_seq(midi_data, index_2)), out_path + str(count) + "_1.mid")
    create_file(merge(elim_double(create_seq(midi_data, index_1)), elim_double(create_seq(midi_data, index_2))), out_path + str(count) + "_2.mid")


def separate_files(in_path, out_path, log_name):
    count = 0
    log_list = []
    for file in glob.glob(in_path + "/*.mid"):
        midi_data = pretty_midi.PrettyMIDI(file)
        log_list.append([count, str(file), str(midi_data.instruments)])
        print(log_list[-1])
        create_file(elim_double(create_seq(midi_data, 0)), out_path + str(count) + "_0.mid")
        create_file(elim_double(create_seq(midi_data, 1)), out_path + str(count) + "_1.mid")
        create_file(merge(elim_double(create_seq(midi_data, 0)), elim_double(create_seq(midi_data, 1))), out_path + str(count) + "_2.mid")
        count += 1
    write_log(log_list, log_name + '_log')


'''
# --> Commands for separating files
separate_files('data/raw/violin_duet', './violins/', './violins/violins')
separate_files('data/raw'alto_saxophone_duet', 'alto_saxs/', './alto_saxs/saxs')
separate_files('data/raw/cello_duet', './cellos/', './cellos/cellos')
separate_files('data/raw/trumpet_duet', './trumpets/', './trumpets/trumpets')
separate_files('data/raw/clarinet_duet', './clarinets/', './clarinets/clarinets')
separate_files('data/raw/tubas_duet', './tubas/', './tubas/tubas')
separate_files('data/raw/french_horn_duet', './french_horns/', './french_horns/french_horns')
separate_files('data/raw/flute_duet', './flutes/', './flutes/flutes')
separate_files('data/raw/guitar_duet', './guitars/', './guitars/guitars')
separate_files('data/raw/trombone_duet', './trombones/', './trombones/trombones')
'''
