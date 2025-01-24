import numpy as np 
import sys; sys.path.append('../../interesting_code_snippets/')
from itertools import chain
import time
from functions import read_dx_grid
import pdb


if __name__=="__main__":
    #  if len(sys.argv)<3:
    #    print("Usage: python3 this_script in_file1 in_file2")
    
    in_file1=sys.argv[1]
    in_file2=sys.argv[2]
    thresh = float(sys.argv[3])
    
  #  pdb.set_trace()
    potential_grid1, _, _ = read_dx_grid(in_file1)
    potential_grid2, _, _ = read_dx_grid(in_file2)

    potential_grid_comparison = np.absolute(potential_grid1-potential_grid2)
    
    potential_grid_diff = potential_grid_comparison[potential_grid_comparison>thresh]

    positions = np.argwhere(potential_grid_comparison>0.01)

    print("The number of points with different values is: ", 
          potential_grid_diff.size)

    #print("The positions where the difference occured is: ", positions)
    



