import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Get the correct path to result.csv
SCRIPTS_DIR = Path(__file__).resolve().parent
RESULT_CSV_PATH = SCRIPTS_DIR / "result.csv"

pathway_info=[]
with open(RESULT_CSV_PATH) as f:
    for index, lines in enumerate(f):
        if index == 0:
            continue

        else:
            tokens = lines.split(",")
            pathway_info.append(tokens[2:])

pathway_info = np.reshape(pathway_info,(37,len(pathway_info[0])))
pathway_info = pathway_info.astype(np.int64)
