import numpy as np
from scipy.interpolate import *
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt
import sys


def apply_stencil(u):
    i, j, k = u.shape
    result = np.zeros_like(u)
    result[1:-1, 1:-1, 1:-1] = (
        + 6*u[1:-1, 1:-1, 1:-1]
        - u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]
        - u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]
        - u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]
        
    )
    return result

def compute_res(u, rhs):
    #breakpoint()
    temp = apply_stencil(u)-rhs
    return np.sqrt(np.sum(temp*temp)/ temp.size)

def rb_gauss_seidel_3d(u, f, iterations=3):
    n, m, p = u.shape
    for _ in range(iterations):
        # Red update: (i + j + k) % 2 == 0
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                for k in range(1, p - 1):
                    if (i + j + k) % 2 == 0:
                        u[i, j, k] = (1/6) * (
                            u[i+1, j, k] + u[i-1, j, k] +
                            u[i, j+1, k] + u[i, j-1, k] +
                            u[i, j, k+1] + u[i, j, k-1] +
                            f[i, j, k]
                        )
        # Black update: (i + j + k) % 2 == 1
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                for k in range(1, p - 1):
                    if (i + j + k) % 2 == 1:
                        u[i, j, k] = (1/6) * (
                            u[i+1, j, k] + u[i-1, j, k] +
                            u[i, j+1, k] + u[i, j-1, k] +
                            u[i, j, k+1] + u[i, j, k-1] +
                            f[i, j, k]
                        )
    return u


def restrict(res):
    return res[::2, ::2, ::2]



def prolong_with_rgi(u_coarse):
    nx, ny, nz = u_coarse.shape

    # 1. Define the coarse grid coordinates
    x_coarse = np.linspace(0, 1, nx)
    y_coarse = np.linspace(0, 1, ny)
    z_coarse = np.linspace(0, 1, nz)

    # 2. Set up the interpolator
    interpolator = RegularGridInterpolator(
        (x_coarse, y_coarse, z_coarse),
        u_coarse,
        method='linear',
        bounds_error=False,
        fill_value=0.0  # Can also be 'nearest' or extrapolation value
    )

    # 3. Define the fine grid coordinates (twice as fine)
    nx_f, ny_f, nz_f = 2 * nx - 1, 2 * ny - 1, 2 * nz - 1
    x_fine = np.linspace(0, 1, nx_f)
    y_fine = np.linspace(0, 1, ny_f)
    z_fine = np.linspace(0, 1, nz_f)

    # 4. Create meshgrid and points to evaluate
    X, Y, Z = np.meshgrid(x_fine, y_fine, z_fine, indexing='ij')
    points = np.stack((X, Y, Z), axis=-1)  # shape (nx_f, ny_f, nz_f, 3)

    # 5. Interpolate
    u_fine = interpolator(points)  # returns shape (nx_f, ny_f, nz_f)

    return u_fine


def set_boundary_conditions(u):
    i, j, k = np.shape(u)
    for x in range(i):
        for y in range(j):
            u[x, y, k-1]=1
            u[x, y, 0]= 0

    for x in range(i):
        for z in range(k):
            u[x, j-1, z] = z / (k-1)
            u[x, 0, z] = z / (k-1)

    for y in range(j):
        for z in range(k):
            u[i-1, y, z] = z / (k-1)
            u[0, y, z] = z / (k-1)



# Matrix-vector product for flattened input
def matvec_shape(x_flat, shape):
    x_grid = np.zeros([shape[0]+2,shape[1]+2,shape[2]+2])
    x_grid[1:-1, 1:-1, 1:-1] = x_flat.reshape(shape)
    Ax_grid = apply_stencil(x_grid) / 4  #We need to divide by 4 in order to account for different grid steps
    return Ax_grid[1:-1, 1:-1, 1:-1].ravel()

def get_matvec_shape(shape):
    return lambda x_flat: matvec_shape(x_flat, shape)

def multigrid_2_V_cycle(base_length, num_iters):
    level_1_length = 2*base_length+1
    level_2_length = 2*2*base_length+1

    #setting boundary conditions
    u = np.zeros(level_2_length)
    set_boundary_conditions(u)

    #print(u)

    #Allocating right hand side
    rhs = -apply_stencil(u)

    #Allocating working array
    guess = np.zeros(level_2_length)

    #getting matvec function and linear operator
    matvec = get_matvec_shape(level_1_length-2)
    n = (2*base_length[0]-1)*(2*base_length[1]-1)*(2*base_length[2]-1)
    A = LinearOperator((n, n), matvec=matvec)

    #Now doing the multigrid operation
    residuals=[]
    for i in range(num_iters):
        #breakpoint()
        residuum = compute_res(guess, rhs)
        print("The residuum after ", i, " iterations is ", residuum)
        residuals.append(residuum)
        rb_gauss_seidel_3d(guess, rhs, iterations=2)
        #print("The guess after a ", 2, " Gauss-Seidel iteration is: ")
        #print(guess)
        #coarsening
        defect_domain = rhs-apply_stencil(guess)
        #print("The defect domain before coarsening is: ")
        #print(defect_domain)
        coarse = restrict(defect_domain)
        #print("The right hand side after coarsening is: ")
        #print(coarse)
        #unravel coarse
        unravel_c = coarse[1:-1, 1:-1, 1:-1].ravel()
        #solving coarse_grid
        sol = cg(A, unravel_c)[0]
        #print (sol)
        sol_shaped = sol.reshape(2*base_length-1)

        coarse[1:-1, 1:-1, 1:-1] = sol_shaped

        #print("The solution after the coarse grid solver is:")
        #print(coarse)

        

        #refining
        refinemend = prolong_with_rgi(coarse)

        #print("The refined coarse grid solution: ")
        #print(refinemend)

        #Adding correction
        #print (refinemend.shape)
        guess[1:-1, 1:-1, 1:-1] += refinemend[1:-1, 1:-1, 1:-1]

        #print("After adding the coarse grid correction the guess is:")
        #print(guess)

        

        #doing the post smoothing
        rb_gauss_seidel_3d(guess, rhs, iterations=2)

    #print("After the post smoothing step the guess is:")
    #print(guess)

    guess+=u


    return guess, residuals


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    base_length = np.array([1, 1, 1])
    num_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    guess, residuals = multigrid_2_V_cycle(base_length, num_iters)
   # print(len(sys.argv))



