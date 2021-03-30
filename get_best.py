'''
From a directory of word logs get the best word
Also outputs file with top k words
'''

import argparse
import sys
import os
import scandir

class best_words:
    def __init__(self, num_words):
        self.words = [['none', 0]]*num_words

    def check_word_to_be_added(self, y_avg):
        if y_avg > self.words[-1][1]:
            return True
        else:
            return False

    def add_word(self, word, y_avg):
        self.words.append([word, y_avg])
        # Sort from highest to lowest y_avg
        self.words = sorted(self.words, reverse = True, key = lambda x: x[1])
        # Drop the worst extra word
        self.words = self.words[:-1]

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DIR', type=str, help='Specify directory with all word log files')
    commandLineParser.add_argument('OUTPUT', type=str, help='Specify output file')
    commandLineParser.add_argument('--k', type=int, default=100, help="Specify num_words to keep")

    args = commandLineParser.parse_args()
    words_dir = args.DIR
    output_file = args.OUTPUT
    num_words = args.k

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/get_best.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    best = best_words(num_words)

    # Get list of files in directory
    files = [f.name for f in scandir.scandir(words_dir)]

    for curr_file in files:
        #print("Processing " + curr_file)
        curr_path = words_dir+"/"+curr_file
        with open(curr_path, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            items = line.split()
            word = str(items[0])
            cos_dist = float(items[1])
            #cos_dist = float(items[1][7:-1])
            if best.check_word_to_be_added(abs(cos_dist)):
                best.add_word(word, abs(cos_dist))

    print(best.words)

    # Write words to output file
    with open(output_file, 'w') as f:
        f.write('')
    for item in best.words:
        word = item[0]
        with open(output_file, 'a') as f:
            f.write('\n'+word)
