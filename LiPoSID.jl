module LiPoSID

using LinearAlgebra
using QuantumOptics
using DynamicPolynomials
# using MomentTools
using MosekTools
using Random
using JuMP
using NLopt
using TSSOS
using Clustering
using HDF5

function hankel(y::AbstractArray)
    m, time_duration = size(y) # m - dimention of output vector y, time_duration - length of timeseries (number of time steps)
    q = Int(round(time_duration/2)) # q - is the size of Hankel matrix 
    H = zeros(eltype(y), q * m , q) 
    for r = 1:q, c = 1:q # r - rows, c -columns
        H[(r-1)*m+1:r*m, c] = y[:, r+c-1]
    end
    return H, m
end

function lsid_ACx0(Y::AbstractArray, Δt) #, δ = 1e-6)
    # y - output time series dim[y] = m x number_of_time_steps
    # δ - precission cutoff all the smaller values of Σ will be discarded 
    
    H, m = hankel(Y) # Hankel matrix and dimention of output (should be 12 in our case)
    U, Σ, Vd = svd(H) # Singular value decomposition of H to U,  Σ,  V†
    
    s = Diagonal(sqrt.(Σ)) # Matrix square root 
    U = U * s
    Vd = s * Vd
     
    # n = argmin(abs.(Σ/maximum(Σ) .- δ)) - 1 # estimated rank of the system

    Sigma_log = log.(Σ/maximum(Σ))
    Sigma2D = reshape(Sigma_log, (1, length(Sigma_log)))

    n = minimum(counts(kmeans(Sigma2D, 2))) + 1
    
    C = U[1:m, 1:n] # m -dimention of output, n - rank of the system
    
    U_up = U[1:end-m, :] # U↑
    U_down = U[m+1:end, :] # U↓
    
    A = pinv(U_up) * U_down
    # Ac = log(A)/Δt 
    # Ac = Ac[1:n, 1:n] 
    A = A[1:n, 1:n] # n - estimated rank of the system
    
    x0 = pinv(U) * H
    x0 = x0[1:n, 1]
    
    return A, C, x0 # was A, Ac, C, x0

end

function lsid_n_ACx0(Y::AbstractArray, Δt, n) 
    # y - output time series dim[y] = m x number_of_time_steps
    # n - rank of the system we want to identify
    
    H, m = hankel(Y) # Hankel matrix and dimention of output (should be 12 in our case)
    U, Σ, Vd = svd(H) # Singular value decomposition of H to U,  Σ,  V†
    
    s = Diagonal(sqrt.(Σ)) # Matrix square root 
    U = U * s
    Vd = s * Vd
      
    C = U[1:m, 1:n] # m -dimention of output, n - rank of the system
    
    U_up = U[1:end-m, :] # U↑
    U_down = U[m+1:end, :] # U↓
    
    A = pinv(U_up) * U_down
    # Ac = log(A)/Δt 
    # Ac = Ac[1:n, 1:n] 
    A = A[1:n, 1:n] # n - estimated rank of the system
    
    x0 = pinv(U) * H
    x0 = x0[1:n, 1]
    
    return A, C, x0 # was A, Ac, C, x0

end

function propagate(A, C, x0, steps)
    n = size(A, 1)
    @assert size(x0,1) == n
    y = zeros(size(C,1), steps) 
    xₜ = x0
    for t in 1:steps
        y[:, t] = C * xₜ
        xₜ = A * xₜ
    end
    return y
end 

function rand_dm(n)
    # return a random density matrix
    ρ = -1 .+ 2 * rand(n, n) 
    ρ += im * (-1 .+ 2 * rand(n, n))  
    ρ = ρ * ρ'
    Hermitian(ρ / tr(ρ))
end

function rand_herm(n)
    # return a random hermitian matrix
    h = -1 .+ 2 * rand(n, n)
    h += im *(-1 .+ 2 *  rand(n, n))
    h = 0.5 * (h + h')
    Hermitian(h)
end

function bloch(ρ_list::Matrix{ComplexF64})
    # Pauli matricies
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]   
    bloch_vec = [real(tr(σ[i] * ρ_list[t])) for i=1:3]
end

function bloch(ρ_list::Vector{Matrix{ComplexF64}})
    # Pauli matricies
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    time_steps = length(ρ_list)
    bloch_vec = zeros(3, time_steps)
    for t in 1:time_steps
        bloch_vec[:, t] = [real(tr(σ[i] * ρ_list[t])) for i=1:3]
    end
    bloch_vec
end

function bloch(ρ_list::Vector{Hermitian{ComplexF64, Matrix{ComplexF64}}})
    # Pauli matricies
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    time_steps = length(ρ_list)
    bloch_vec = zeros(3, time_steps)
    for t in 1:time_steps
        bloch_vec[:, t] = [real(tr(σ[i] * ρ_list[t])) for i=1:3]
    end
    bloch_vec
end

function rho_from_bloch(bloch_vec::Vector{Float64})
    # Pauli matricies
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    ρ = (sum([bloch_vec[i] * σ[i] for i=1:3]) + I)/2 
    ρ ::Matrix{ComplexF64}
end

function rho_series_from_bloch(bloch_vec::Matrix{Float64})
    time_steps = size(bloch_vec, 2)
    ρ = Vector{Matrix{ComplexF64}}() # size !!!
    for t in 1:time_steps
        push!(ρ, rho_from_bloch(bloch_vec[:, t]))     
    end
    ρ ::Vector{Matrix{ComplexF64}}
end

function rho3d_from_bloch(bloch_vec::Matrix{Float64})
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    time_steps = size(bloch_vec, 2)
    ρ = zeros(2, 2, time_steps) + im*zeros(2, 2, time_steps)
    for t in 1:time_steps
        ρ[:, :, t] = (sum([bloch_vec[i, t] * σ[i] for i=1:3]) + I)/2       
    end
    ρ ::Array{ComplexF64, 3}
end

function rand_Linblad_w_noise(basis, seed, w, t_list)
    # seed - to generate reproducable system,
    # w - noise level
    # t_list - time span
    
    # basis = NLevelBasis(2) # define 2-level basis

    Random.seed!(seed)    
    
    ρ₀ = DenseOperator(basis, rand_dm(2))  # initial state density matrix
    H = DenseOperator(basis, rand_herm(2)) # Hamiltonian of the system
    J = DenseOperator(basis, (-1 .+ 2 *randn(2, 2)) + im*(-1 .+ 2 *randn(2, 2))) # Lindblad decipator  was rand !!!!!!
    
    time, ρ_exact = timeevolution.master(t_list, ρ₀, H, [J])

    ρ = [ (1 - w) * ρₜ.data + w * rand_dm(2) for ρₜ in ρ_exact ], H.data, J.data
       
end

function frobenius_norm2(m)
    return tr(m * m')
end

function lindblad_rhs(ρ, H, J::Matrix)
    """
    Right hand side of the Lindblad master equation
    """
    return -im * (H * ρ - ρ * H) + J * ρ * J' - (J' * J  * ρ + ρ * J' * J) / 2
    
end

function lindblad_rhs(ρ, H, J::Array)
    """
    Right hand side of the Lindblad master equation
    """
   
    Σ = sum([ ( Jⱼ * ρ * Jⱼ' - (Jⱼ' * Jⱼ  * ρ + ρ * Jⱼ' * Jⱼ)/2 ) for Jⱼ in J ])
    
    return -im * (H * ρ - ρ * H) + Σ 
    
end

import Base.real
function real(p::AbstractPolynomial)
    sum(real(coef) * mon for (coef, mon) in zip(coefficients(p), monomials(p))) #if ~isapproxzero(abs(coef)))
end

function pade_obj(ρ::Array{ComplexF64,3}, t, H, J)
   
    obj = 0
    for i in 2:size(ρ,3)
        obj += frobenius_norm2(
            ρ[:, :, i] - ρ[:, :, i-1] 
            - (t[i]-t[i-1])*lindblad_rhs((ρ[:, :, i]+ρ[:, :, i-1])/2, H, J)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function pade_obj(ρ::Vector{Matrix{ComplexF64}}, t::Vector{Float64}, H, J)

    obj = 0
    for i in 2:size(ρ,1)
        obj += frobenius_norm2(
            ρ[i] - ρ[i-1] 
            - (t[i]-t[i-1])*lindblad_rhs((ρ[i]+ρ[i-1])/2, H, J)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function simpson_obj(ρ::Vector{Matrix{ComplexF64}}, t, H, J)
    
    obj = 0
    for i in 3:length(ρ)
        obj += frobenius_norm2(
            ρ[i] - ρ[i-2] - (t[i]-t[i-1])lindblad_rhs((ρ[i-2] + 4ρ[i-1] + ρ[i])/3, H, J)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function simpson_obj(ρ::Array{ComplexF64,3}, t, H, J)  
    
    obj = 0
    for i in 3:length(ρ)
        obj += frobenius_norm2(
            ρ[:, :, i] - ρ[:, :, i-2] - (t[i]-t[i-1])lindblad_rhs((ρ[:, :, i-2] + 4ρ[:, :, i-1] + ρ[:, :, i])/3, H, J)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function kraus_obj(ρ::Vector{Matrix{ComplexF64}}, K1, K2) 
    obj = 0
    for i in 1:length(ρ)-1
        obj += frobenius_norm2(K1 * ρ[i] * K1' - ρ[i+1]) + frobenius_norm2(K2 * ρ[i] * K2' - ρ[i+1])
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function kraus_obj_constr(ρ, K) 
    obj = 0
    for i in 1:length(ρ)-1
        obj += frobenius_norm2(sum(k * ρ[i] * k' for k in K) - ρ[i+1])
    end
    constr = frobenius_norm2(sum(k' * k for k in K) - I)
    return real(obj), real(constr)*1e3
end

function timeevolution_kraus(t_steps, ρ₀, K)
    ρ = [ρ₀]
    for t = 2:t_steps
        #push!(ρ, Hermitian(sum([K[i]* ρ[end] * K[i]' for i = 1:length(K)])))
        ρ_next = sum([K[i]* ρ[end] * K[i]' for i = 1:length(K)])
        push!(ρ, ρ_next/tr(ρ_next))
    end
    return ρ
end  

function rand_Kraus_w_noise(seed, w, time_span)
    Random.seed!(seed)
    
    ρ₀ = LiPoSID.rand_dm(2)     

    K1 = rand(2,2) + im*rand(2,2)
    K2 = rand(2,2) + im*rand(2,2)
    
    ρ_exact = timeevolution_kraus(time_span, ρ₀, [K1, K2])
    
    ρ = [ (1 - w) * ρₜ + w * LiPoSID.rand_dm(2) for ρₜ in ρ_exact ]
end

function rand_Kraus_w_noise(seed, w, time_span, kraus_rank)
    Random.seed!(seed)
    
    ρ₀ = LiPoSID.rand_dm(2)
    
    K = [rand(2,2) + im*rand(2,2) for i in 1:kras_rank]
    
    ρ_exact = timeevolution_kraus(time_span, ρ₀, K)
    
    ρ = [ (1 - w) * ρₜ + w * LiPoSID.rand_dm(2) for ρₜ in ρ_exact ] # adding white noise
end


# using NLopt

function minimize_local(obj, constr, guess) # polynomial objective, and guess x candidate
    vars = variables(obj)
    
    @assert length(vars) == length(guess)

    function g(a...)
        # Converting polynomial expression to function to be minimize
        obj(vars => a)
    end
    
    model = Model(NLopt.Optimizer)

    set_optimizer_attribute(model, "algorithm", :LD_MMA)
    
    #set_silent(model)
    @variable(model, y[1:length(vars)]);
    
    for (var, init_val) in zip(y, guess)
        set_start_value(var, init_val)
    end
    
    #= if length(constr_list) > 0
        @constraint(model, constr_list)
    end =# 
    
    register(model, :g, length(y), g; autodiff = true)
    @NLobjective(model, Min, g(y...))
    JuMP.optimize!(model)
    solution = vars => map(value, y)
    
    return solution
end 

function minimize_global(obj, constr_list = [])
    optimizer = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
    obj_min, M = minimize(obj, constr_list, [], variables(obj), maxdegree(obj) ÷ 2, optimizer)
    
    r = get_minimizers(M)
    obj_min_vals = [obj(r[:,i]) for i=1:size(r)[2]]
    best_candidate = r[:, argmin(obj_min_vals)]
    
    minimize_local(obj, constr_list, best_candidate) 
   
end 

# using QuantumOptics

function quantum_series(basis, ρ)
    [ DenseOperator(basis, Hermitian(ρ[i])) for i = 1:length(ρ) ]
end

function fidelity_series(basis, ρ₁, ρ₂)

    @assert  length(ρ₁) == length(ρ₂)

    len_of_series = min(length(ρ₁), length(ρ₂))

    ρ₁ = quantum_series(basis, ρ₁)
    ρ₂ = quantum_series(basis, ρ₂)

    return [abs(fidelity(ρ₁[i], ρ₂[i])) for i in 1:len_of_series]

end

function min_fidelity_between_series(basis, ρ1, ρ2)

    len_of_series = length(ρ1)

    @assert  length(ρ2) == len_of_series

    ρ1q = quantum_series(basis, ρ1)
    ρ2q = quantum_series(basis, ρ2)
    
    minimum([abs(fidelity(ρ1q[i], ρ2q[i])) for i in 1:len_of_series])

end

#using TSSOS

function min2step(obj, constr)
    # obj - is objective function
    # constr - one constraint in the form of equation
    
    # extract valiables from the objective
    vars = variables(obj)

    iter = 0
    best_sol = ones(length(vars))
    
    # Perform global minimization with TSSOS package
    try
        opt,sol,data = tssos_first([obj, constr], variables(obj), maxdegree(obj)÷2, numeq=1, solution=true, QUIET = true); 
    
        # execute higher levels of the TSSOS hierarchy
        iter = 1
        best_sol = sol

        while ~isnothing(sol)
            iter += 1
            best_sol = sol
            try
                opt,sol,data = tssos_higher!(data, solution=true, QUIET = true);
            catch
                break
            end
            
            if iter > 5
                best_sol = ones(length(vars))
                break
            end
        end
    catch
        best_sol = ones(length(vars))
    end  
   
    function g(a...)
        # Converting polynomial expression of objective to function to be minimized
        obj(vars => a)
    end
    
    function e(a...)
        # Converting polynomial expression of constraint to function to be minimize
        constr(vars => a)
    end
       
    # Create NLopt model
    model = Model(NLopt.Optimizer)

    # Set algorithm 
    set_optimizer_attribute(model, "algorithm", :LD_SLSQP) 
    
    # Set variables
    @variable(model, y[1:length(vars)]);

    # Register constraint
    register(model, :e, length(y), e; autodiff = true)
    
    @NLconstraint(model, e(y...) == 0)

    # Register objective
    register(model, :g, length(y), g; autodiff = true)
    @NLobjective(model, Min, g(y...))
    
    # Set guess
    guess = best_sol
    for (var, init_val) in zip(y, guess)
        set_start_value(var, init_val)
    end

    # Call JuMP optimization function
    JuMP.optimize!(model)

    solution = vars => map(value, y)

    return(solution, iter)
end  

function min2step(obj)
    # obj - is objective function

    # extract valiables from the objective
    vars = variables(obj)
    
    # Perform global minimization with TSSOS package
    iter = 0
    best_sol = ones(length(vars))

    try
        opt,sol,data = tssos_first(obj, variables(obj), solution=true, QUIET = true);
        # execute higher levels of the TSSOS hierarchy
        iter = 1
        best_sol = sol

        while ~isnothing(sol)
            iter += 1
            best_sol = sol
            try
                opt,sol,data = tssos_higher!(data, solution=true, QUIET = true);
            catch
                break
            end
            
            if iter > 5
                best_sol = ones(length(vars))
                break
            end
        end
    catch
        best_sol = ones(length(vars))
    end  
    
   
    function g(a...)
        # Converting polynomial expression of objective to function to be minimized
        obj(vars => a)
    end
       
    # Create NLopt model
    model = Model(NLopt.Optimizer)

    # Set algorithm 
    set_optimizer_attribute(model, "algorithm", :LD_SLSQP) 
    
    # Set variables
    @variable(model, y[1:length(vars)]);

    # Register objective
    register(model, :g, length(y), g; autodiff = true)
    @NLobjective(model, Min, g(y...))
    
    # Set guess
    guess = best_sol
    for (var, init_val) in zip(y, guess)
        set_start_value(var, init_val)
    end

    # Call JuMP optimization function
    JuMP.optimize!(model)

    solution = vars => map(value, y)

    return(solution, iter)
end  

#### HDF5 READING RESULTS ####

function get_seeds_and_timespan(file_name)   
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        seeds = read(fid["seeds"])
        Δt = read(fid["dt"])
        tₘₐₓ = read(fid["t_max"])
        return seeds,  Δt, tₘₐₓ
    end
end

function get_noise_levels(file_name)   
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        noise_levels = keys(fid["data_by_noise_level"])
        return noise_levels
    end
end

function get_variable_names(file_name, noise_level, seed)
        h5open(file_name,"r") do fid   # read file, preserve existing contents
        variable_names = keys(fid["data_by_noise_level"][string(noise_level)][string(seed)])
        return variable_names
    end
end

function get_by_name(file_name, var_name, noise_levels, seeds)
        h5open(file_name,"r") do fid # read file, preserve existing contents
        var_by_name = []
        for w in noise_levels
            current_noise_var = [ read(fid["data_by_noise_level"][string(w)][string(seed)][var_name]) for seed in seeds ]
            push!(var_by_name, current_noise_var)
        end
        return(var_by_name)
    end
end

function get_lsid(file_name, noise, seeds)
    A = get_by_name(file_name, "A", [noise], seeds)[1]
    C = get_by_name(file_name, "C", [noise], seeds)[1]
    x0 = get_by_name(file_name, "x0", [noise], seeds)[1]
    return A, C, x0
end

function get_kraus_sid(file_name, noise, seeds)  
    K1_sid = get_by_name(file_name, "K1_sid", [noise], seeds)[1]
    K2_sid = get_by_name(file_name, "K2_sid", [noise], seeds)[1]
    return K1_sid, K2_sid
end 

function get_lindblad_params(file_name, noise, key,  seeds, basis)
    H = [DenseOperator(basis, Hl) for Hl in get_by_name(file_name, "H_"*key, [noise], seeds)[1]]
    J = [DenseOperator(basis, Jl) for Jl in get_by_name(file_name, "J_"*key, [noise], seeds)[1]]
   return H, J
end

function lindblad_evolution(key, time_limit, Δt, noise_level, seed)
    time_span = [0:Δt:time_limit;]
    H_exact = DenseOperator(basis, get_by_name(file_name, "H_"*key, [noise_level], seed)[1][1])
    J_exact = DenseOperator(basis, get_by_name(file_name, "J_"*key, [noise_level], seed)[1][1])
    ρ0 = DenseOperator(basis, get_by_name(file_name, "rho0", [noise_level], seed)[1][1])
    time, ρ_exact_ser  = timeevolution.master(time_span, ρ0, H_exact, [J_exact])
    ρ = [ρₜ.data for ρₜ in ρ_exact_ser]
end

function lindblad_evolution_data(time_span, ρ0, H, J)
    time, ρ_ser  = timeevolution.master(time_span, ρ0, H, [J])
    ρ = [ρₜ.data for ρₜ in ρ_ser]
end


function read_fidelity_table(file_name, fid_name, noise, seeds)
    fidelity_table = []
    h5open(file_name,"r") do fid # read file, preserve existing contents
        for seed in seeds
            push!(fidelity_table, read(fid[string(noise)][string(seed)][string(fid_name)]))
        end
        return(mapreduce(permutedims, vcat, fidelity_table))
    end
end

end
