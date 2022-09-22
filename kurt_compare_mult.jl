
using HDF5

# File to save results to HDF5
res_file_name = "Kurt_compare_methods_3_operators_" * string(Dates.format(now(), "yyyy-u-dd_at_HH-MM")) * ".h5"

# Names of files to read data from
basis_file_names = ["State_B"*string(n) for n=1:4]
dodeca_file_names = ["State_D"*string(n) for n=1:20]

data_file_names = [basis_file_names; dodeca_file_names] # vcat(basis_file_names, dodeca_file_names)

@time for df in data_file_names

    # read all the data from one dodeca-data file

    data_file_name = "/home/zah/PycharmProjects/Kurt2021/2022JAN24/DATA/"*df*"_data.h5"
    ρ = []
    t = []
    for γⱼ in γ  
        ρⱼ, tⱼ = get_rho_series(data_file_name, string(γⱼ))
        push!(ρ,ρⱼ)
        push!(t,tⱼ)
    end 

    h5open(res_file_name,"cw") do fid   # read-write, create file if not existing, preserve existing contents
        create_group(fid, string(df))
    end

    lck_write = ReentrantLock() # create lock to use when writing in loop below

    Threads.@threads for i in length(γ)

        #create symbolic operators

        @polyvar x[1:4]
        H_symb = [ 1.0 * x[1]              x[3] + im * x[4]
                   x[3] - im * x[4]        x[2]             ]

        @polyvar ja1[1:2, 1:2]
        @polyvar jb1[1:2, 1:2]
        J1_symb = 1.0 * ja1 + im * jb1

        @polyvar ja2[1:2, 1:2]
        @polyvar jb2[1:2, 1:2]
        J2_symb = 1.0 * ja2 + im * jb2

        @polyvar a1[1:2, 1:2]
        @polyvar b1[1:2, 1:2]
        K1_symb = 1.0 * a1 + im * b1

        @polyvar a2[1:2, 1:2]
        @polyvar b2[1:2, 1:2]
        K2_symb = 1.0 * a2 + im * b2

        @polyvar a3[1:2, 1:2]
        @polyvar b3[1:2, 1:2]
        K3_symb = 1.0 * a3 + im * b3

        K_symb_list = [K1_symb, K2_symb, K3_symb]

        # get exact data series

        ρᵢ = convert(Vector{Matrix{ComplexF64}}, ρ[i])
        tᵢ = convert(Vector{Float64}, t[i])
        Δt = tᵢ[2]-tᵢ[1]
        time_steps = length(tᵢ)

        # IDENTIFICATION

        # Polynomial objectives with (1) Pade and (2) Simpsom methods
        obj_pade = LiPoSID.pade_obj(ρᵢ, tᵢ, H_symb, J_symb)
        obj_simp = LiPoSID.simpson_obj(ρᵢ, tᵢ, H_symb, J_symb)

        solution_pade, tssos_iter_pade = LiPoSID.min2step(obj_pade)
        solution_simp, tssos_iter_simp = LiPoSID.min2step(obj_simp)

        H_sid_pade = subs(H_symb, solution_pade)
        J_sid_pade = subs(J_symb, solution_pade)
        ρ_sid_pade = Lindblad_time_evolution(basis, ρᵢ[1], tᵢ, H_sid_pade, J_sid_pade)

        H_sid_simp = subs(H_symb, solution_simp)
        J_sid_simp = subs(J_symb, solution_simp)
        ρ_sid_simp = Lindblad_time_evolution(basis, ρᵢ[1], tᵢ, H_sid_simp, J_sid_simp)

        fidelity_pade = LiPoSID.min_fidelity_between_series(basis, ρ_sid_pade, ρᵢ)
        fidelity_simp = LiPoSID.min_fidelity_between_series(basis, ρ_sid_simp, ρᵢ)

        # Polynomial objective and constrain assuming (3) Kraus evolution
        obj_kraus, constr_kraus = LiPoSID.kraus_obj_constr(ρᵢ, K1_symb, K2_symb)
        solution_kraus, tssos_iter_kraus = LiPoSID.min2step(obj_kraus, constr_kraus)
        K1_sid = subs(K1_symb, solution_kraus)
        K2_sid = subs(K2_symb, solution_kraus)
        ρ_sid_kraus = LiPoSID.timeevolution_kraus(time_steps, ρᵢ[1], [K1_sid, K2_sid])
        fidelity_kraus = LiPoSID.min_fidelity_between_series(basis, ρ_sid_kraus, ρᵢ)

        # (4) Linear SID as benckmark
  
        A, C, x0 = LiPoSID.lsid_ACx0(LiPoSID.bloch(ρᵢ), Δt)
        bloch_sid = LiPoSID.propagate(A, C, x0, time_steps)
        ρ_lsid = LiPoSID.rho_series_from_bloch(bloch_sid)
        fidelity_lsid = LiPoSID.min_fidelity_between_series(basis, ρ_lsid, ρᵢ)

        # Saving results to HDF5

        

        lock(lck_write)
        
        try
            
            h5open(res_file_name,"cw") do fid  # read-write, create file if not existing, preserve existing contents

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
            
        finally
            unlock(lck_write)
        end

    end # of multi-thread loop over γ

end # of loop over the dodecahedron states files