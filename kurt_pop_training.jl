include("LiPoSID.jl")

using Dates
using QuantumOptics
using LinearAlgebra
using HDF5

using Suppressor

# Auxilary functions to read Kurt's data 

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


# MAIN

date_and_time_string =  string(Dates.format(now(), "yyyy-u-dd_at_HH-MM"))

println()
println("Polynomial System Identification for two level open bosonic system (Kurt simulated data)")

# File to save results to HDF5
res_file_name = "Kurt_POPSID_operators_" * date_and_time_string * ".h5"

println("Results to be saved in HDF5 file: "*res_file_name)

# Names of files with Kurt data
basis_file_names = ["State_B"*string(n) for n=1:4]
dodeca_file_names = ["State_D"*string(n) for n=1:20]
# data_file_names = [basis_file_names; dodeca_file_names] # vcat(basis_file_names, dodeca_file_names)
train_files = basis_file_names

operators_num = 4 # number of Kraus operators 

# Read available coupling levels
df = train_files[1]
# full_data_file_name = "/home/zah/PycharmProjects/Kurt2021/2022JAN24/DATA/"*df*"_data.h5"
full_data_file_name = "C:/Users/Zakhar/Documents/GitHub/JPOP_SID/DATA/"*df*"_data.h5"
γ = get_keys(full_data_file_name) # noise levels 
println("Available coupling levels gamma in ", γ)

# γ = ["79.477", "251.33"]

# Create symbolic operators

@polyvar h[1:4]
H_symb = [ 1.0 * h[1]              h[3] + im * h[4]
           h[3] - im * h[4]        h[2]             ]

@polyvar ja[1:2, 1:2, 1:operators_num-1]
@polyvar jb[1:2, 1:2, 1:operators_num-1]

J_symb = []

for i in 1:operators_num-1
    J = ja[:,:,i] + im * jb[:,:,i]
    push!(J_symb, J)
end

@polyvar ka[1:2, 1:2, 1:operators_num]
@polyvar kb[1:2, 1:2, 1:operators_num]

K_symb = []

for i in 1:operators_num
    K = ka[:,:,i] + im * kb[:,:,i]
    push!(K_symb, K)
end

basis = NLevelBasis(2)

# Assembling one objective for each noise level from all files
println("Assembling polynomial objectives and constrains:")

@time for γᵢ in  γ   # loop over γ coupling (noise) levels   
    
    println("gamma =", γᵢ) 
 
    obj_pade = 0
    obj_simp = 0
    obj_kraus = 0
    constr_kraus = 0

    for df in train_files # loop over initial states

        print("    Processing file: ", df)
        
        # Read data series of Kurt data

        ρ, t = get_rho_series(full_data_file_name, string(γᵢ))

        ρ = convert(Vector{Matrix{ComplexF64}}, ρ)
        t = convert(Vector{Float64}, t)

        Δt = t[2]-t[1]
        time_steps = length(t)

        print(" Assembling objective... ")

        print(" Pade... ")
        # Polynomial objectives with (1) Pade and (2) Simpsom methods
        obj_pade += LiPoSID.pade_obj(ρ, t, H_symb, J_symb)
        print(" Simpson... ")
        obj_simp += LiPoSID.simpson_obj(ρ, t, H_symb, J_symb)
        
        print(" Kraus... ")
        # Polynomial objective and constrain assuming (3) Kraus evolution
        objk, constrk = LiPoSID.kraus_obj_constr(ρ, K_symb)
        obj_kraus += objk 
        constr_kraus += constrk #!!!!

        println("done.")

    end # of files (initial states) loop  

    println("Performing system identification with polynomial optimization ...")

    # Polynomial optimization assuming Lindblad evolution

    print(" Pade: ")
    solution_pade, tssos_iter_pade = LiPoSID.min2step(obj_pade)
    println(" Number of TSSOS iterations: ", tssos_iter_pade)

    print(" Simpson: ")
    solution_simp, tssos_iter_simp = LiPoSID.min2step(obj_simp)
    println(" Number of TSSOS iterations: ", tssos_iter_simp)

    H_sid_pade = subs(H_symb, solution_pade)
    J_sid_pade =  [ subs(J_symb, solution_pade)  for J_symb in J_symb_list ]
    
    H_sid_simp = subs(H_symb, solution_simp)
    J_sid_simp = [ subs(J_symb, solution_simp) for J_symb in J_symb_list ] 

    # Polynomial optimization assuming Kraus evolution
    print(" Kraus: ")
    solution_kraus, tssos_iter_kraus = LiPoSID.min2step(obj_kraus, constr_kraus)
    println(" Number of TSSOS iterations: ", tssos_iter_kraus)
    K_sid = [ subs(K_symb, solution_kraus)  for K_symb in K_symb_list ]

    println("optimization done.")

    print("Saving for gamma[i] ...")

    # Save results to HDF5     

    h5open("C:/Users/Zakhar/Documents/GitHub/JPOP_SID/DATA/"*res_file_name,"cw") do fid  # read-write, create file if not existing, preserve existing contents

        γ_group = create_group(fid, "gamma_"*string(γᵢ))

        γ_group["H_sid_pade"] = convert.(ComplexF64, H_sid_pade)
        γ_group["J1_sid_pade"] = convert.(ComplexF64, J_sid_pade[1])
        γ_group["J2_sid_pade"] = convert.(ComplexF64, J_sid_pade[2])

        γ_group["H_sid_simp"] = convert.(ComplexF64, H_sid_simp)
        γ_group["J1_sid_simp"] = convert.(ComplexF64, J_sid_simp[1])
        γ_group["J2_sid_simp"] = convert.(ComplexF64, J_sid_simp[2])

        γ_group["K1_sid"] = convert.(ComplexF64, K_sid[1])
        γ_group["K2_sid"] = convert.(ComplexF64, K_sid[2])
        γ_group["K3_sid"] = convert.(ComplexF64, K_sid[3])

        γ_group["tssos_iter_pade"] = tssos_iter_pade
        γ_group["tssos_iter_simp"] = tssos_iter_simp
        γ_group["tssos_iter_kraus"] = tssos_iter_kraus

    end # of HDF5 writing

    println("done.")
    
end # of  γ coupling (noise) levels loop 


# Identified operators are saved in HDF5 file for futher processing.

