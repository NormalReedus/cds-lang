# Week 6 assignment - Networks

After downloading or cloning this repo `cd` into week6 (this dir):
```bash
cd week6
```

## Virtual Environment
To create a virtual environment, follow the instructions for your system.

### MacOS / Linux
`week6/`
```bash
bash create_network_venv.sh
```

### Windows
`week6/`
```bash
bash create_network_win_venv.sh
```
Your terminal should now create the venv, install dependencies from requirements.txt, and activate the environment. Your terminal should show that you are in the `network_venv`.

## Running the script
The script takes 3 arguments:
1. `-p` or `--path` [**OBLIGATORY**]: The filepath to the edgelist you wish to read (I have provided an `edgelist.csv` in this directory taken from class).
2. `-w` or `--min-weight` [OPTIONAL]: The minimum amount of edge weight to include (defaults to 0 if not passed).
3. `-d` or `--demo` [OPTIONAL]: Whether or not to only work on a subset of 10k edges, since the script can be really slow (defaults to false of not passed).

### Examples
`week6/`
```bash
# Run on everything
python network.py -p edgelist.csv

# Filter off low-weight edges
python network.py -p edgelist.csv -w 300

# Only work on a subset and filter off low-weight edges
python network.py -p edgelist.csv -w 300 -d
```

## Output
The vizualisation will be saved in `./viz/network.png` and the output data in `./output/measures.csv` **according to where you ran the script from**, so I recommend `cd`ing into `week6/` before running the script.
