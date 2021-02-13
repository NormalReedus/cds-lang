#!../lang101/Scripts/python
import argparse # Used for passing arguments from the command line (optional part of assignment)
from pathlib import Path # Used to force the format of the passed-in file-directory (optional part of assignment)
import os # Path concatenation
import csv # For writing the output file
import re # For tokenization

def get_files(dir):
	pass

def calc_collocates(keyword): # See https://www.english-corpora.org/mutualInformation.asp and http://www.collocations.de/AM/index.html
	pass

def tokenize(input_string):
    # Split on any non-alphanumeric character
    tokenizer = re.compile(r"\W+")
    
    # Tokenize 
    token_list = tokenizer.split(input_string)

    return token_list


def main(file_dir, keyword, window_size):
	outfile = os.path.join('.', 'out', f'{keyword}.csv')

	# Loop through texts and read them
	# Tokenize text into list
	# Save N as the total number of tokens
	# Find indices in the token list of the tokens that match the keyword
	# Slice +- window size around matching indices
	# Add all of the tokens in the slice to a list of collocates - we now have a list of all the tokens that appear within `window_size` of the keyword
	# Calculate the other values - TBD






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Calculate collocates for a specific keyword.')
	parser.add_argument('file_dir', type=Path, help='the directory containing all of your text files to analyze.')
	parser.add_argument('keyword', help='the keyword to look for.')
	parser.add_argument('window_size', type=int, nargs="?", default=10, help='the number of words on both sides of the keyword to look for collocates in.')
	args = parser.parse_args()	

	main(args.file_dir, args.keyword, args.window_size)