#!../lang101/Scripts/python
import argparse # Used for passing arguments from the command line (optional part of assignment)
from pathlib import Path # Used to force the format of the passed-in file-directory (optional part of assignment)
import os # Path concatenation
import csv # For writing the output file

def get_files(dir):
	pass

def cal_collocates(keyword):
	pass


def main(file_dir, keyword, window_size):
	outfile = os.path.join('.', 'out', f'{keyword}.csv')






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Calculate collocates for a specific keyword.')
	parser.add_argument('file_dir', type=Path, help='the directory containing all of your text files to analyze.')
	parser.add_argument('keyword', help='the keyword to look for.')
	parser.add_argument('window_size', type=int, nargs="?", default=10, help='the number of words on both sides of the keyword to look for collocates in.')
	args = parser.parse_args()	

	main(args.file_dir, args.keyword, args.window_size)