import numpy as np
import sys; sys.path.append('../../interesting_code_snippets/')
from itertools import chain
import time 
from functions import read_dx_grid
import pdb

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python3 this_script in_file1")
    
    in_file1=sys.argv[1]
    potential_grid1, _, _ = read_dx_grid(in_file1)

    for idx in range(potential_grid1.shape[2]):
        plane = potential_grid1[:, :, idx]
        for idx in range(plane.shape[1]):
            line = plane[:, idx]
            print(line)
        print("")


