#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.interpolate import *
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from operator import *


# In[6]:


def apply_stencil(u, diag = 1, should_print = True):
    #if should_print:
    #    print("The factor diag in apply stencil is: ", diag)
    i, j, k = u.shape
    result = np.zeros_like(u)
    result[1:-1, 1:-1, 1:-1] = (
        + 6*u[1:-1, 1:-1, 1:-1]
        - u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]
        - u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]
        - u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]
        
    ) / diag
    return result

def compute_res(u, rhs, diag):
    #print("The factor diag in compute res is: ", diag)
    temp = apply_stencil(u, diag) - rhs
    return np.sqrt(np.sum(temp*temp)) / np.sqrt(np.sum(rhs*rhs))



def rb_gauss_seidel_3d(u, f, diag=1, iterations=3):
    #print("The factor diag in rb_gauss_seidel_3d is: ", diag)
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
                            diag * f[i, j, k]
                        )
        # Black update: (i + j + k) % 2 == 1
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                for k in range(1, p - 1):
                    if (i + j + k) % 2 == 1:
                        u[i, j, k] =  (1/6) * (
                            u[i+1, j, k] + u[i-1, j, k] +
                            u[i, j+1, k] + u[i, j-1, k] +
                            u[i, j, k+1] + u[i, j, k-1] +
                            diag * f[i, j, k]
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


# In[7]:


def analytic_u(x, y, z):
    return -4*x**2-3*y**2+7*z**2


def set_boundary_conditions(u):
    i, j, k = np.shape(u)
    for x in range(i):
        for y in range(j):
            u[x, y, k-1]=analytic_u(x/(i-1), y/(j-1), 1)
            u[x, y, 0]= analytic_u(x/(i-1), y/(j-1), 0)

    for x in range(i):
        for z in range(k):
            u[x, j-1, z] = analytic_u(x/(i-1), 1, z / (k-1))
            u[x, 0, z] = analytic_u(x/(i-1), 0, z / (k-1))

    for y in range(j):
        for z in range(k):
            u[i-1, y, z] = analytic_u(1, y/(j-1), z / (k-1))
            u[0, y, z] = analytic_u(0, y/(j-1), z / (k-1))



shape=(3,3,3)

# Matrix-vector product for flattened input
def matvec_shape(x_flat, shape, factor = 2):
    x_grid = np.zeros([shape[0]+2,shape[1]+2,shape[2]+2])
    x_grid[1:-1, 1:-1, 1:-1] = x_flat.reshape(shape)
    Ax_grid = apply_stencil(x_grid, should_print = False) / factor #We need to divide by 2 in order to get good convergence
    return Ax_grid[1:-1, 1:-1, 1:-1].ravel()

def get_matvec_shape(shape, factor=2):
    #print("The factor in get_matvec is: ", factor)
    return lambda x_flat: matvec_shape(x_flat, shape, factor)


# In[8]:


def multigrid_V_cycle(base_length, num_iters=[1 , 1], num_smooth = 2, num_levels = None, level_num = None, rhs = None):

    if (num_levels==None or level_num==None):
        num_levels=level_num=len(num_iters)

    if(len(num_iters) == 1):
        num_iters = [num_iters[0] for i in range(num_levels)]
    
    length_by_level = [2**n*base_length + 1 for n in range(1, level_num + 1)]

    #Allocating working array
    guess = np.zeros(length_by_level[level_num-1])
    u = np.zeros(length_by_level[level_num-1])

    #if in uppermost level, set up the problem
    if rhs is None:        
        #setting boundary conditions       
        set_boundary_conditions(u)
        #print(u)
        #Allocating right hand side
        rhs = -apply_stencil(u)



    #getting matvec function and linear operator
    if level_num == 1:
        matvec = get_matvec_shape(length_by_level[0]-2, 2**(num_levels-level_num))
        n = (2*base_length[0]-1)*(2*base_length[1]-1)*(2*base_length[2]-1)
        A = LinearOperator((n, n), matvec=matvec)
        unravel_c = rhs[1:-1, 1:-1, 1:-1].ravel()
        #solving coarse_grid
        sol = cg(A, unravel_c)[0]
        #print (sol)
        sol_shaped = sol.reshape(2*base_length-1)
        rhs[1:-1, 1:-1, 1:-1] = sol_shaped
        
        return rhs




    #Now doing the multigrid operation
    residuals = []
    for i in range(num_iters[0]):
        if level_num==num_levels:
            residuum = compute_res(guess, rhs, 2**(num_levels-level_num))
            print("After ", i, " iterations the residual is ", residuum)
            residuals.append(residuum)

        #print("The current level is: ", level_num)
        rb_gauss_seidel_3d(guess, rhs, 2**(num_levels-level_num), iterations = num_smooth)
        #print("The guess after the rb_guass_seidel is: ")
        #print(guess)
        #print("The right hand side for the ")
        #print("The guess after a ", 2, " Gauss-Seidel iteration is: ")
        #print(guess)
        #coarsening
        defect_domain = rhs-apply_stencil(guess, diag = 2**(num_levels-level_num))
        #print("The defect domain before coarsening is: ")
        #print(defect_domain)
        coarse = restrict(defect_domain)
        coarse = multigrid_V_cycle(base_length, num_iters[1 : ] ,num_smooth ,num_levels, level_num-1,rhs = coarse)

        #refining
       
        refinemend = prolong_with_rgi(coarse)

        #print("The refined coarse grid solution: ")
        #print(refinemend)

        #Adding correction
        #print (refinemend.shape)
        guess[1:-1, 1:-1, 1:-1]+=refinemend[1:-1, 1:-1, 1:-1]

        #print("After adding the coarse grid correction the guess is:")
        #print(guess)
        
        #doing the post smoothing
        #print("Current level: ", level_num)
        rb_gauss_seidel_3d(guess, rhs, 2**(num_levels-level_num), iterations= num_smooth)

        #print("After the post smoothing step the guess is:")
        #print(guess)

    guess+=u


    return (guess, residuals) if level_num==num_levels else guess


# In[9]:


np.set_printoptions(precision=3)
base_length = np.array([16,16,16])
#

#multigrid_V_cycle(base_length, [10, 1], num_levels = 3, level_num=2);
guess, residuals = multigrid_V_cycle(base_length, [10, 1]);

xs= np.linspace(0, 9, 10)

popt, pcov  = curve_fit(lambda b, a: pow(a, b), xs, residuals, p0 = [0.05])

plt.xlabel("num iters")
plt.ylabel("Residual")
plt.title("Residual after n iterations")
plt.yscale("log")
plt.grid(True, which="both", linestyle='--')
plt.plot(xs, residuals)
plt.savefig("Residual_plot_assymetric.pdf",format="pdf")

print("The computed convergence rate is: ", popt[0])


# In[7]:


np.set_printoptions(precision=3)
base_length = np.array([16,16,16])
guess, residuals = multigrid_V_cycle(base_length, [10, 1, 1], 2)

xs= np.linspace(0, 9, 10)

popt, pcov  = curve_fit(lambda b, a: pow(a, b), xs, residuals, p0 = [0.05])
print("The computed convergence rate is: ", popt[0])

plt.xlabel("num iters")
plt.ylabel("Residual")
plt.title("Residual after n iterations")
plt.yscale("log")
plt.grid(True, which="both", linestyle='--')
plt.plot(xs, residuals)
plt.savefig("Residual_plot_3_l_assymetric.pdf",format="pdf")


# In[8]:


np.set_printoptions(precision=3)
base_length = np.array([16,16,16])
guess, residuals = multigrid_V_cycle(base_length, [10, 1, 1, 1], 2)

popt, pcov  = curve_fit(lambda b, a: pow(a, b), xs, residuals, p0 = [0.05])
print("The computed convergence rate is: ", popt[0])

xs= np.linspace(0, 9, 10)
plt.xlabel("num iters")
plt.ylabel("Residual")
plt.title("Residual after n iterations")
plt.yscale("log")
plt.grid(True, which="both", linestyle='--')
plt.plot(xs, residuals)
plt.savefig("Residual_plot_4_l_assymetric.pdf",format="pdf")


# In[ ]:


num_iters = 10

np.set_printoptions(precision=3)
base_length = np.array([16, 16, 16])
guess, residuals = multigrid_V_cycle(base_length, [num_iters, 1, 1, 1, 1], 2)

xs= np.linspace(0, num_iters-1, num_iters)

popt, pcov  = curve_fit(lambda b, a: pow(a, b), xs, residuals, p0 = [0.05])
print("The computed convergence rate is: ", popt[0])

plt.xlabel("num iters")
plt.ylabel("Residual")
plt.title("Residual after n iterations")
plt.yscale("log")
plt.grid(True, which="both", linestyle='--')
plt.plot(xs, residuals)
plt.savefig("Residual_plot_5_l_assymetric.pdf",format="pdf")


# In[168]:


num_iters = 10

np.set_printoptions(precision=3)
base_length = np.array([1,1,1])
guess, residuals = multigrid_V_cycle(base_length, [num_iters, 1, 1, 1, 1, 1])

popt, pcov  = curve_fit(lambda b, a: pow(a, b), xs, residuals, p0 = [0.05])
print("The computed convergence rate is: ", popt[0])

xs= np.linspace(0, num_iters-1, num_iters)
plt.xlabel("num iters")
plt.ylabel("Residual")
plt.title("Residual after n iterations")
plt.yscale("log")
plt.grid(True, which="both", linestyle='--')
plt.plot(xs, residuals)
plt.savefig("Residual_plot_6_l_assymetric.pdf",format="pdf")

