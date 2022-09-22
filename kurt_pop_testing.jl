include("LiPoSID.jl")

using Dates
using QuantumOptics
using LinearAlgebra
using HDF5

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

# function to produce time series of density matrices 

function Lindblad_time_evolution(basis, ρ₀, time_span, H, J)
         
    ρ₀ = DenseOperator(basis, Hermitian(ρ₀)) 
    H = DenseOperator(basis, H) # reconstructed Hamiltonian of the system
    J = [ DenseOperator(basis, Jᵢ) for Jᵢ in J ] # reconstracted Lindblad decipators
    
    time, ρ  = timeevolution.master(time_span, ρ₀, H, J)
    
    ρ = [ρₜ.data for ρₜ in ρ]

end

function get_operator(file, group, operator_name)

    h5open(file,"r") do fid # read-only
        A = read(fid[group][operator_name])
        return A
    end
end

# MAIN

println()
println()
println("Testing the POP SID on decahedron initial states (Kurt simulated data)")

date_and_time_string =  string(Dates.format(now(), "yyyy-u-dd_at_HH-MM"))

# Define quantum basis with QuantumOptics package
basis = NLevelBasis(2)

# Names of files with Kurt data
train_files = basis_file_names = ["State_B"*string(n) for n=1:4]
test_files = dodeca_file_names = ["State_D"*string(n) for n=1:20]

pop_sid_ops_file = "Kurt_POPSID_operators_2022-Sep-21_at_00-09.h5"
res_fidelity_file_name = "Kurt_POPSID_fidelity_dodeca_"*date_and_time_string*".h5"

directory = "C:/Users/Zakhar/Documents/GitHub/JPOP_SID/DATA/"

println("Results of fidelity calculations to be saved in HDF5 file: "*res_fidelity_file_name)

# Read available coupling levels

γ = get_keys(directory*pop_sid_ops_file) # noise levels 

γ = [string(chop(γᵢ, head = 6, tail = 0)) for γᵢ in γ]

println("Available coupling levels γ ∈ ", γ)

@time for γᵢ in  γ  # loop over γ coupling (noise) levels

    println("γ =", γᵢ) 

    # Create gamma coupling group in output HDF5 file 

    h5open(directory*res_fidelity_file_name,"cw") do fid   # read-write, create file if not existing, preserve existing contents
        γ_group = create_group(fid, "gamma_"*string(γᵢ))
    end 

    # Read operators identified with POP SID from HDF5 file

    H_pade = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"H_sid_pade")
    J1_pade = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"J1_sid_pade")
    J2_pade = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"J2_sid_pade")
    J_pade = [J1_pade, J2_pade]

    H_simp = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"H_sid_simp")
    J1_simp = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"J1_sid_simp")
    J2_simp = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"J2_sid_simp") 
    J_simp = [J1_simp, J2_simp]

    K1 = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"K1_sid")
    K2 = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"K2_sid")
    K3 = get_operator(directory*pop_sid_ops_file, "gamma_"*string(γᵢ),"K3_sid")
    K = [K1, K2, K3]

    # Loop over initial states 

    print("    Processing files: ")

    for df in test_files # loop over initial states

        print(df*" ")
        
        # Read EXACT data series of Kurt data

        ρ, t = get_rho_series(directory*df*"_data.h5", string(γᵢ))

        ρ = convert(Vector{Matrix{ComplexF64}}, ρ)
        t = convert(Vector{Float64}, t)

        Δt = t[2]-t[1]
        t_steps = length(t)

        # Restore data series POP identified

        ρ₀ = ρ[1]

        ρ_pade = Lindblad_time_evolution(basis, ρ₀, t, H_pade, J_pade)
        ρ_simp = Lindblad_time_evolution(basis, ρ₀, t, H_simp, J_simp)

        ρ_kraus = LiPoSID.timeevolution_kraus(t_steps, ρ₀, K)

        # Calculating fidelity series

        fid_pade = LiPoSID.fidelity_series(basis, ρ, ρ_pade)
        fid_simp = LiPoSID.fidelity_series(basis, ρ, ρ_simp)
        fid_kraus = LiPoSID.fidelity_series(basis, ρ, ρ_kraus)

        # Saving fidelity series for each initial state 

        h5open(directory*res_fidelity_file_name,"cw") do fid   # read-write, create file if not existing, preserve existing contents
            γ_group = open_group(fid, "gamma_"*string(γᵢ)) # open coupling group
            init_state_group = create_group(γ_group, string(df)) # create initial state group
            init_state_group["fidelity_simp"] = convert.(Float64, fid_simp)
            init_state_group["fidelity_pade"] = convert.(Float64, fid_pade)
            init_state_group["fidelity_kraus"] = convert.(Float64, fid_kraus)
        end

    end # of files (initial states) df loop  

    println()

end # of γ coupling (noise) levels loop 

