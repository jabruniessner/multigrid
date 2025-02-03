import numpy as np 
import sys; sys.path.append('../../interesting_code_snippets/')
from itertools import chain
import time
from functions import read_dx_grid
import pdb


if __name__=="__main__":
    if len(sys.argv)<4:
        print("Usage: python3 this_script in_file1 in_file2 thresh")
    
    in_file1=sys.argv[1]
    in_file2=sys.argv[2]
    thresh = float(sys.argv[3])
    
  #  pdb.set_trace()
    potential_grid1, _, _ = read_dx_grid(in_file1)
    potential_grid2, _, _ = read_dx_grid(in_file2)

    potential_grid_comparison = np.absolute(potential_grid1-potential_grid2)
    
    #potential_grid_diff = potential_grid_comparison[potential_grid_comparison>thresh]

    #positions = np.argwhere(potential_grid_comparison>0.01)

    #print("The number of points with different values is: ", 
    #      potential_grid_diff.size)
    
    print("The maximal difference is given by", np.max(potential_grid_comparison))
    maximum=np.max(potential_grid_comparison)
    maximum_pos=np.argwhere(potential_grid_comparison==maximum)[0]
    print("The maximum position is ", maximum_pos)
    print("The value in the first grid is: ", potential_grid1[maximum_pos[0], maximum_pos[1], maximum_pos[2]])
    print("The value in the second grid is: ", potential_grid2[maximum_pos[0], maximum_pos[1], maximum_pos[2]])

    #print("The values are: ", potential_grid_diff)

    #print("The positions where the difference occured is: ", positions)

    #a = potential_grid1.flatten()
    #b = potential_grid2.flatten()

    #print("First grid:      Second grid")
    #for idx, _  in enumerate(a):
    #    print(a[idx], "       ", b[idx])
    



