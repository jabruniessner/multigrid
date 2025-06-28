using NNlib
using PyCall
using IterativeSolvers
using LinearMaps
np=pyimport("numpy")

function apply_stencil(u, diag = 1)
    i, j, k = np.shape(u)
    result = np.zeros_like(u)
    result[2:end-1, 2:end-1, 2:end-1] = (
        + 6*u[2:end-1, 2:end-1, 2:end-1]
        - u[3:end, 2:end-1, 2:end-1] - u[1:end-2, 2:end-1, 2:end-1]
        - u[2:end-1, 3:end, 2:end-1] - u[2:end-1, 1:end-2, 2:end-1]
        - u[2:end-1, 2:end-1, 3:end] - u[2:end-1, 2:end-1, 1:end-2]
        
    ) / diag
    return result
end


function compute_res(u, rhs, diag)
    #print("The factor diag in compute res is: ", diag)
    temp = apply_stencil(u, diag) .- rhs
    return sum(temp .* temp) / sqrt(sum(rhs .* rhs))
end


function rb_gauss_seidel_3d(u, f; diag=1, iterations=3)
    #print("The factor diag in rb_gauss_seidel_3d is: ", diag)
    n, m, p = np.shape(u)
    for _ = 1:iterations
        # Red update: (i + j + k) % 2 == 0
        for i = 2:n-1
            for j = 2: m - 1
                for k = 2: p - 1
                    if (i + j + k) % 2 == 1
                        u[i, j, k] = (1/6) * (
                            u[i+1, j, k] + u[i-1, j, k] +
                            u[i, j+1, k] + u[i, j-1, k] +
                            u[i, j, k+1] + u[i, j, k-1] +
                            diag * f[i, j, k]
                        )

                    end
                end 
            end
        end
        # Black update: (i + j + k) % 2 == 1
        for i=2:n-1
            for j=2:m-1
                for k=2:p - 1
                    if (i + j + k) % 2 == 0
                        u[i, j, k] =  (1/6) * (
                            u[i+1, j, k] + u[i-1, j, k] +
                            u[i, j+1, k] + u[i, j-1, k] +
                            u[i, j, k+1] + u[i, j, k-1] +
                            diag * f[i, j, k]
                        )
                    end
                end
            end
        end
    end
end


function restrict(res)
    return res[1:2:end, 1:2:end, 1:2:end]
end

function prolong(coarse)
    i, j, k = np.shape(coarse)
    temp = reshape(coarse, (size(coarse)..., 1, 1))
    temp2 = upsample_trilinear(temp; size=(2*i-1, 2*j-1, 2*k-1))
    
    return reshape(temp2, (2*i-1, 2*j-1, 2*k-1))
end

function set_boundary_conditions(u)
    i, j, k = np.shape(u)
    for x =1:i
        for y = 1:j
            u[x, y, end]=1
            u[x, y, 1] = 0
        end
    end

    for x =1:i
        for z =1:k
            u[x, end, z] = (z-1) / (k-1)
            u[x, 1, z] = (z-1) / (k-1)
        end 
    end

    for y = 1:j
        for z = 1:k
            u[end, y, z] = (z-1) / (k-1)
            u[1, y, z] = (z-1) / (k-1)
        end
    end
end

function matvec_shape(x_flat, shape; factor=2)
    x_grid = np.zeros((shape[1]+2, shape[2]+2, shape[3]+1))
    x_grid[2:end-1, 2:end-1, 2:end-1] = reshape(x_flat, shape)
    Ax_grid = apply_stencil(x_grid) ./ factor
    return vec(Ax_grid)
end

u=np.zeros([5,5,5])
set_boundary_conditions(u)
rhs = -apply_stencil(u)
rhs_flat = reshape(rhs[2:end-1, 2:end-1, 2:end-1], :)

linear_function = x_flat -> matvec_shape(x_flat, (3,3,3); factor = 1)

A_op = FunctionMap(x_flat -> matvec_shape(x_flat, (3,3,3); factor = 1), 3*3*3)
guess = np.zeros_like(u)

rb_gauss_seidel_3d(guess, rhs, diag=1, iterations=3)

new = restrict(guess)
refined = prolong(new)

vec(new)


