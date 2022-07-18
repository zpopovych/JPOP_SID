include("LiPoSID.jl")
using Random
Random.seed!(140722)

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

function get_rho_series(file_name, γ)
    h5open(file_name, "r") do file
        ρt = read(file[string(γ)])
        t = ρt["t"]
        ρ00 = ρt["p0"]; ρ01re = ρt["s_re"];  ρ01im = ρt["s_im"]
        ρ = []
        for i in 1:length(t)
            ρ_i = [ ρ00[i]                    ρ01re[i] + im * ρ01im[i]
                    ρ01re[i] - im * ρ01im[i]  1 - ρ00[i]               ]
            push!(ρ, ρ_i)
        end
        return(ρ, t)
    end
end

using DynamicPolynomials
using QuantumOptics


# MAIN LOOP

using HDF5
# File to save results to HDF5
res_file_name = "Kurt_clust_compare_methods_started_" * string(Dates.format(now(), "yyyy-u-dd_at_HH-MM")) * ".h5"

basis_file_names = ["State_B"*string(n)*"_data.h5" for n=1:4]
dodeca_file_names = ["State_D"*string(n)*"_data.h5" for n=1:20]
# data_file_names = [basis_file_names; dodeca_file_names] # vcat(basis_file_names, dodeca_file_names)

data_file_names = ["State_B1_data.h5"]

@time for df in data_file_names
    println("Processing file: ", df)
    w_list = h5open(df, "r") do file
    return keys(file)
    end

    h5open(res_file_name,"cw") do fid   # read-write, create file if not existing, preserve existing contents
            create_group(fid, sting(df))
        end

    for w in w_list

        # Create noise level group
        h5open(res_file_name,"cw") do fid   # read-write, create file if not existing, preserve existing contents
            create_group(fid[sting(df)], string(w))
        end

        println("w=", w)

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

        # Read data series of Kurt data

        full_data_file_name = "/home/zah/PycharmProjects/Kurt2021/2022JAN24/DATA/"*df

        ρ, t = get_rho_series(full_data_file_name, string(w))
        Δt = t[2]-t[1]
        time_steps = length(t)

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

            noise_level_group = open_group(fid[string(df)], string(w))

            noise_level_group["rho0"] = ρ[1]

            noise_level_group["H_sid_pade"] = convert.(ComplexF64, H_sid_pade)
            noise_level_group["J_sid_pade"] = convert.(ComplexF64, J_sid_pade)

            noise_level_group["H_sid_simp"] = convert.(ComplexF64, H_sid_simp)
            noise_level_group["J_sid_simp"] = convert.(ComplexF64, J_sid_simp)

            noise_level_group["K1_sid"] = convert.(ComplexF64, K1_sid)
            noise_level_group["K2_sid"] = convert.(ComplexF64, K2_sid)

            noise_level_group["A"] = A
            noise_level_group["C"] = C
            noise_level_group["x0"] = x0

            noise_level_group["fidelity_pade"] = fidelity_pade
            noise_level_group["fidelity_simp"] = fidelity_simp
            noise_level_group["fidelity_kraus"] = fidelity_kraus
            noise_level_group["fidelity_lsid"] = fidelity_lsid

            noise_level_group["tssos_iter_pade"] =  tssos_iter_pade
            noise_level_group["tssos_iter_simp"] =  tssos_iter_simp
            noise_level_group["tssos_iter_kraus"] =  tssos_iter_kraus
        end
    end
end
