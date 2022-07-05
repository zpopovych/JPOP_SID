module LiPoSID

using LinearAlgebra
using QuantumOptics
using DynamicPolynomials, MomentTools
using MosekTools
using Random
using JuMP
using NLopt

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

function rand_Linblad_w_noise(seed, w, t_list)
    # seed - to generate reproducable system,
    # w - noise level
    # t_list - time span
    
    Random.seed!(seed)    
    
    ρ₀ = DenseOperator(basis, rand_dm(2))  # initial state density matrix
    H = DenseOperator(basis, rand_herm(2)) # Hamiltonian of the system
    J = DenseOperator(basis, (-1 .+ 2 *randn(2, 2)) + im*(-1 .+ 2 *randn(2, 2))) # Lindblad decipator  was rand !!!!!!
    
    time, ρ_exact = timeevolution.master(t_list, ρ₀, H, [J])

    ρ = [ (1 - w) * ρₜ.data + w * rand_dm(2) for ρₜ in ρ_exact ];
    
    bloch(ρ)    
end

function frobenius_norm2(m)
    return tr(m * m')
end

function lindblad_rhs(ρ, H, J)
    """
    Right hand side of the Lindblad master equation
    """
    return -1im * (H * ρ - ρ * H) + J * ρ * J' - (J' * J  * ρ + ρ * J' * J) / 2
    
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

function pade_obj(ρ::Vector{Matrix{ComplexF64}}, t, H, J)
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

function kraus_obj(ρ, K1, K2) 
    obj = 0
    for i in 1:length(ρ)-1
        obj += frobenius_norm2(K1 * ρ[i] * K1' - ρ[i+1]) + frobenius_norm2(K2 * ρ[i] * K2' - ρ[i+1])
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

# using NLopt

function minimize_local(obj, guess) # polynomial objective, and guess x candidate
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
    
    register(model, :g, length(y), g; autodiff = true)
    @NLobjective(model, Min, g(y...))
    JuMP.optimize!(model)
    solution = vars => map(value, y)
    
    return solution
end 

function minimize_global(obj)
    optimizer = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
    obj_min, M = minimize(obj, [], [], variables(obj), maxdegree(obj) ÷ 2, optimizer)
    
    r = get_minimizers(M)
    obj_min_vals = [obj(r[:,i]) for i=1:size(r)[2]]
    best_candidate = r[:, argmin(obj_min_vals)]
    
    minimize_local(obj, best_candidate) 
   
end 

end
