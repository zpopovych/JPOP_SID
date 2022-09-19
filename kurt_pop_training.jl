include("LiPoSID.jl")
using Random
Random.seed!(140722)

using Dates
using QuantumOptics
using LinearAlgebra

function Lindblad_time_evolution(basis, ρ₀, time_span, H, J)
         
    ρ₀ = DenseOperator(basis, Hermitian(ρ₀)) 
    H = DenseOperator(basis, H) # reconstructed Hamiltonian of the system
    J = [ DenseOperator(basis, Jᵢ) for Jᵢ in J ] # reconstracted Lindblad decipators
    
    time, ρ  = timeevolution.master(time_span, ρ₀, H, J)
    
    ρ = [ρₜ.data for ρₜ in ρ]

end

function get_rho_series(file_name, γ)
    h5open(file_name, "r") do file
        ρᵧ = read(file[string(γ)])
        t = ρᵧ["t"]
        ρ₀₀ = ρᵧ["p0"]; Re_ρ₀₁ = ρᵧ["s_re"];  Im_ρ₀₁ = ρᵧ["s_im"]
        ρ_series = []
        t_series = []

        for i in 1:length(t)
            ρᵢ= [ ρ₀₀[i]                      Re_ρ₀₁[i] + im * Im_ρ₀₁[i]
                  Re_ρ₀₁[i] - im * Im_ρ₀₁[i]  1 - ρ₀₀[i]                 ]
            push!(ρ_series, convert(Matrix{ComplexF64}, ρᵢ))
            push!(t_series, convert(Float64, t[i]))
        end
        return(ρ_series, t_series)
    end
end

function get_keys(df)
    h5open(df, "r") do file
        return keys(file)
    end
end

using DynamicPolynomials
using QuantumOptics


# MAIN LOOP

using HDF5
# File to save results to HDF5
res_file_name = "Kurt_POPSID_operators_" * string(Dates.format(now(), "yyyy-u-dd_at_HH-MM")) * ".h5"

# Names of files with Kurt data
basis_file_names = ["State_B"*string(n) for n=1:4]
dodeca_file_names = ["State_D"*string(n) for n=1:20]
# data_file_names = [basis_file_names; dodeca_file_names] # vcat(basis_file_names, dodeca_file_names)
train_files = basis_file_names

operator_rank = 3 # number of Kraus operators 

# Read available coupling levels
df = train_files[1]
full_data_file_name = "/home/zah/PycharmProjects/Kurt2021/2022JAN24/DATA/"*df*"_data.h5"
γ = get_keys(full_data_file_name) # noise levels 
println("Available coupling levels γ ∈ ", γ)

obj_pade = Array{AbstractPolynomial}(undef, length(γ))
obj_simp = Array{AbstractPolynomial}(undef, length(γ))
obj_kraus = Array{AbstractPolynomial}(undef, length(γ))
constr_kraus = Array{AbstractPolynomial}(undef, length(γ))

# Create symbolic operators

@polyvar x[1:4]
H_symb = [ 1.0 * x[1]              x[3] + im * x[4]
           x[3] - im * x[4]        x[2]             ]

@polyvar ja1[1:2, 1:2]
@polyvar jb1[1:2, 1:2]
J1_symb = 1.0 * ja1 + im * jb1

@polyvar ja2[1:2, 1:2]
@polyvar jb2[1:2, 1:2]
J2_symb = 1.0 * ja2 + im * jb2

J_symb_list = [J1_symb, J2_symb]

@polyvar ka1[1:2, 1:2]
@polyvar kb1[1:2, 1:2]
K1_symb = 1.0 * ka1 + im * kb1

@polyvar ka2[1:2, 1:2]
@polyvar kb2[1:2, 1:2]
K2_symb = 1.0 * ka2 + im * kb2

@polyvar ka3[1:2, 1:2]
@polyvar kb3[1:2, 1:2]
K3_symb = 1.0 * ka3 + im * kb3

K_symb_list = [K1_symb, K2_symb, K3_symb]

basis = NLevelBasis(2)

# Assembling one objective for each noise level from all files
println("Assembling polynomial objectives and constrains...")

@time for df in train_files   # loop over initial states
    println("Processing file: ", df)

    for i in 1:length(γ) # loop pver coupling levels 

        println("γ =", γ[i])     

        # Read data series of Kurt data

        ρ, t = get_rho_series(full_data_file_name, string(w))

        ρ = convert(Vector{Matrix{ComplexF64}}, ρ)
        t = convert(Vector{Float64}, t)

        Δt = t[2]-t[1]
        time_steps = length(t)

        # Polynomial objectives with (1) Pade and (2) Simpsom methods
        obj_pade[i] += LiPoSID.pade_obj(ρ, t, H_symb, J_symb)
        obj_simp[i] += LiPoSID.simpson_obj(ρ, t, H_symb, J_symb)

        # Polynomial objective and constrain assuming (3) Kraus evolution
        objk, constrk = LiPoSID.kraus_obj_constr(ρ, K_symb_list)
        obj_kraus[i] += objk
        constr_kraus[i] += constrk

    end # of coupling (noise) levels loop ( variable i )
end # of files (initial states) loop 

# Performing system identification over the assembled polynomial objectives

println("Performing system identification with polynomial optimization ...")

lck_write = ReentrantLock() # create lock to use when writing in loop below

Threads.@threads for i in 1:length(γ) # loop over coupling levels 

    println("γ =", γ[i])   

    # Polynomial optimization assuming Lindblad evolution

    solution_pade, tssos_iter_pade = LiPoSID.min2step(obj_pade[i])
    solution_simp, tssos_iter_simp = LiPoSID.min2step(obj_simp[i]

    H_sid_pade = subs(H_symb, solution_pade)
    J_sid_pade =  [ subs(J_symb, solution_pade)  for J_symb in J_symb_list ]
    ρ_sid_pade = Lindblad_time_evolution(basis, ρ[1], t, H_sid_pade, J_sid_pade)

    H_sid_simp = subs(H_symb, solution_simp)
    J_sid_simp = [ subs(J_symb, solution_simp) for J_symb in J_symb_list ] 
    ρ_sid_simp = Lindblad_time_evolution(basis, ρ[1], t, H_sid_simp, J_sid_simp)


    # Polynomial optimization assuming Kraus evolution
    
    solution_kraus, tssos_iter_kraus = LiPoSID.min2step(obj_kraus[i], constr_kraus[i])
    K_sid = [ subs(K_symb, solution_kraus)  for K_symb in K_symb_list ]
    ρ_sid_kraus = LiPoSID.timeevolution_kraus(time_steps, ρ[1], K_sid)

    # Save results to HDF5

    lock(lck_write)
        
    try

        h5open(res_file_name,"cw") do fid  # read-write, create file if not existing, preserve existing contents

            γ_group = create_group(fid, "gamma_"*string(γ[i]))

            γ_group["H_sid_pade"] = convert.(ComplexF64, H_sid_pade)
            γ_group["J1_sid_pade"] = convert.(ComplexF64, J_sid_pade)
            γ_group["J2_sid_pade"] = convert.(ComplexF64, J_sid_pade)

            γ_group["H_sid_simp"] = convert.(ComplexF64, H_sid_simp)
            γ_group["J1_sid_simp"] = convert.(ComplexF64, J_sid_simp)
            γ_group["J2_sid_pade"] = convert.(ComplexF64, J_sid_pade)

            γ_group["K1_sid"] = convert.(ComplexF64, K1_sid)
            γ_group["K2_sid"] = convert.(ComplexF64, K2_sid)
            γ_group["K3_sid"] = convert.(ComplexF64, K2_sid)

        end # of HDF5 writing
    
    finally
        unlock(lck_write)
    end # of lock structure

end # of coupling (noise) levels loop ( variable i )