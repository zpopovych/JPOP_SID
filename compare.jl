include("LiPoSID.jl")
using Random

Random.seed!(140722)

# n_samples = 100000
n_samples = 1000
seeds = rand(UInt, n_samples)
@assert allunique(seeds)

const tₘₐₓ = 2.0 # maximum time
const Δt = 0.01     # time step
const t = [0:Δt:tₘₐₓ;] # time span
const time_steps = length(t)

using Dates
using QuantumOptics
using LinearAlgebra

function Lindblad_time_evolution(basis, ρ0, time_span, H_sid, J_sid)
         
    ρ0_sid = DenseOperator(basis, Hermitian(ρ0)) 

    H_sid = DenseOperator(basis, H_sid) # reconstructed Hamiltonian of the system
    J_sid = DenseOperator(basis, J_sid) # reconstracted Lindblad decipator
    
    time, ρ_sid_ser  = timeevolution.master(time_span, ρ0_sid, H_sid, [J_sid])
    
    ρ_sid = [ρₜ.data for ρₜ in ρ_sid_ser]

end

using DynamicPolynomials


using QuantumOptics


# MAIN LOOP

const w_list = [0.0 0.01 0.02 0.03 0.04 0.05 0.08 0.1]
#const w_list = [0.0 0.01 0.05]

using HDF5
# File to save results to HDF5
file_name = "LiPoSID_clust_compare_methods_started_" * string(Dates.format(now(), "yyyy-u-dd_at_HH-MM")) * ".h5"
# Save all seeds in separate dataset
h5open(file_name,"cw") do fid   # read-write, create file if not existing, preserve existing contents
    fid["seeds"] = seeds
    fid["dt"] = Δt 
    fid["t_max"] = tₘₐₓ
    create_group(fid, "data_by_noise_level")
end

@time for w in w_list

    # Create noise level group
    
    h5open(file_name,"cw") do fid   # read-write, create file if not existing, preserve existing contents
        create_group(fid["data_by_noise_level"], string(w))
    end
    
    println("w=", w)
       

    Threads.@threads for i=1:n_samples 
        @polyvar x[1:4]
        H_symb = [ 1.0 * x[1]              x[3] + im * x[4]
                   x[3] - im * x[4]        x[2]             ]

        @polyvar a[1:2, 1:2]
        @polyvar b[1:2, 1:2]
        J_symb = 1.0 * a + im * b

        @polyvar a1[1:2, 1:2]
        @polyvar b1[1:2, 1:2]
        K1_symb = 1.0 * a1 + im * b1

        @polyvar a2[1:2, 1:2]
        @polyvar b2[1:2, 1:2]
        K2_symb = 1.0 * a2 + im * b2

        K_symb_list = [K1_symb, K2_symb]

        basis = NLevelBasis(2)

        # println("seed=", seeds[i])

        # Random time series of density matrices using Lindblad master eqution
        ρ, H_exact, J_exact = LiPoSID.rand_Linblad_w_noise(basis, seeds[i], w, t)

        # Polynomial objectives with (1) Pade and (2) Simpsom methods
        obj_pade = LiPoSID.pade_obj(ρ, t, H_symb, J_symb)
        obj_simp = LiPoSID.simpson_obj(ρ, t, H_symb, J_symb)

        solution_pade, tssos_iter_pade = LiPoSID.min2step(obj_pade)
        solution_simp, tssos_iter_simp = LiPoSID.min2step(obj_simp)

        H_sid_pade = subs(H_symb, solution_pade)
        J_sid_pade = subs(J_symb, solution_pade)
        ρ_sid_pade = Lindblad_time_evolution(basis, ρ[1], t, H_sid_pade, J_sid_pade)

        H_sid_simp = subs(H_symb, solution_simp)
        J_sid_simp = subs(J_symb, solution_simp)  
        ρ_sid_simp = Lindblad_time_evolution(basis, ρ[1], t, H_sid_simp, J_sid_simp)

        fidelity_pade = LiPoSID.min_fidelity_between_series(basis, ρ_sid_pade, ρ)
        fidelity_simp = LiPoSID.min_fidelity_between_series(basis, ρ_sid_simp, ρ)     

        # Polynomial objective and constrain assuming (3) Kraus evolution
        obj_kraus, constr_kraus = LiPoSID.kraus_obj_constr(ρ, K1_symb, K2_symb)
        solution_kraus, tssos_iter_kraus = LiPoSID.min2step(obj_kraus, constr_kraus)
        K1_sid = subs(K1_symb, solution_kraus)
        K2_sid = subs(K2_symb, solution_kraus)
        ρ_sid_kraus = LiPoSID.timeevolution_kraus(time_steps, ρ[1], [K1_sid, K2_sid])
        fidelity_kraus = LiPoSID.min_fidelity_between_series(basis, ρ_sid_kraus, ρ)

        # (4) Linear SID as benckmark
        #δ = 1e-2
        A, C, x0 = LiPoSID.lsid_ACx0(LiPoSID.bloch(ρ), Δt) #, δ)
        # A, C, x0 = LiPoSID.lsid_n_ACx0(LiPoSID.bloch(ρ), Δt, 4)
        bloch_sid = LiPoSID.propagate(A, C, x0, time_steps)
        ρ_lsid = LiPoSID.rho_series_from_bloch(bloch_sid)
        fidelity_lsid = LiPoSID.min_fidelity_between_series(basis, ρ_lsid, ρ)

        # Save results to HDF5

        h5open(file_name,"cw") do fid  # read-write, create file if not existing, preserve existing contents
            
            noise_level_group = open_group(fid["data_by_noise_level"], string(w))            
            seed_group = create_group(noise_level_group, string(seeds[i]))

            seed_group["H_exact"] = H_exact
            seed_group["J_exact"] = J_exact
            seed_group["rho0"] = ρ[1]   

            seed_group["H_sid_pade"] = convert.(ComplexF64, H_sid_pade)
            seed_group["J_sid_pade"] = convert.(ComplexF64, J_sid_pade)
            
            seed_group["H_sid_simp"] = convert.(ComplexF64, H_sid_simp)
            seed_group["J_sid_simp"] = convert.(ComplexF64, J_sid_simp)
            
            seed_group["K1_sid"] = convert.(ComplexF64, K1_sid)
            seed_group["K2_sid"] = convert.(ComplexF64, K2_sid)

            seed_group["A"] = A
            seed_group["C"] = C  
            seed_group["x0"] = x0    

            seed_group["fidelity_pade"] = fidelity_pade
            seed_group["fidelity_simp"] = fidelity_simp
            seed_group["fidelity_kraus"] = fidelity_kraus
            seed_group["fidelity_lsid"] = fidelity_lsid
            
            seed_group["tssos_iter_pade"] =  tssos_iter_pade
            seed_group["tssos_iter_simp"] =  tssos_iter_simp
            seed_group["tssos_iter_kraus"] =  tssos_iter_kraus
            
        end
    end
end
