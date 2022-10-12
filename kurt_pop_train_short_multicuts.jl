include("LiPoSID.jl")

using Dates
using DynamicPolynomials
using QuantumOptics
using LinearAlgebra
using HDF5
#using Suppressor

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

#  --- MAIN --- 

date_and_time_string =  string(Dates.format(now(), "yyyy-u-dd_at_HH-MM"))

println("----------------------------------------------------------------------------------------")
println(" Polynomial System Identification for two level open bosonic system (Kurt simulated data)")
println("   Training on 4-BASIS states for γt_cuts = [0.25, 0.5, 1, 2, 4, 8, 16] ")
println("----------------------------------------------------------------------------------------")


# Names of files with Kurt data
basis_file_names = ["State_B"*string(n) for n=1:4]
dodeca_file_names = ["State_D"*string(n) for n=1:20]
# data_file_names = [basis_file_names; dodeca_file_names] # vcat(basis_file_names, dodeca_file_names)
train_files = basis_file_names

directory = "/home/zah/PycharmProjects/JPOP_SID/CUT_DATA/"
# directory = "C:/Users/Zakhar/Documents/GitHub/JPOP_SID/CUT_DATA/"

# Read available coupling levels
df = train_files[1]
# full_data_file_name = "/home/zah/PycharmProjects/Kurt2021/2022JAN24/DATA/"*df*"_data.h5"
first_data_file_name = directory*df*"_CUT_data.h5"
γ = get_keys(first_data_file_name) # noise levels 
println("Available coupling levels gamma in ", γ)

# γ = ["79.477", "251.33"]

lck_read  = ReentrantLock() # create lock to use when reading in the loop below
lck_write = ReentrantLock() # create lock to use when writing in the loop below

# size of the operators is 2
basis = NLevelBasis(2)

# Set γt cuts to be used for all series (all γt cuts for each γ)

γt_cuts = [0.25, 0.5, 1, 2, 4, 8, 16]


@time for operators_num in 2:4 # number of Kraus operators 

    # File to save results to HDF5
    res_file_name = "Kurt_POPSID_MULT_CUT_train-B_ops-" * string(operators_num) * "_" * date_and_time_string * ".h5"

    println("Results to be saved in HDF5 file: "*res_file_name)

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

    
    # Assembling one objective for each noise level from all files
    println("Assembling polynomial objectives and constrains:")


    for γᵢ in  γ   # loop over γ coupling (noise) levels
        
        println("Processing for γ =", γᵢ)

        # Create group in HDF5 file for particular γ

        #lock(lck_write)
                    
        #try

        h5open(directory*res_file_name,"cw") do fid  # read-write, create file if not existing, preserve existing contents

            γ_group = create_group(fid, "gamma_"*string(γᵢ))
            
        end # of HDF5 writing

        #finally
        #    unlock(lck_write)
        #end
        
        for γᵢtₗᵢₘ in γt_cuts # loop over γt cuts [0.25, 0.5, 1, 2, 4, 8, 16]

            obj_pade = 0
            obj_simp = 0
            obj_kraus = 0
            constr_kraus = 0   

            println("Processing for γt cut at ", γᵢtₗᵢₘ)

            for df in train_files # loop over initial states

                full_data_file_name = directory*df*"_CUT_data.h5"

                #print(string(γᵢ)*", file: "*df*" |")
                
                # Read data series of Kurt data

                #lock(lck_read)
                #try
                
                ρ_long, t_long = get_rho_series(full_data_file_name, string(γᵢ))

                #finally
                #    unlock(lck_read)
                #end

                # Prepare for γt cut 
                dt = t_long[2] - t_long[1]
                γᵢt = parse(Float64, γᵢ)*t_long

                if γᵢt[end] >= γᵢtₗᵢₘ - parse(Float64, γᵢ)*dt

                    # Perform γt cut:            
                    step_lim = maximum(findall(x -> x < γᵢtₗᵢₘ, γᵢt))[1]
                    ρ = ρ_long[1:step_lim] 
                    t = t_long[1:step_lim]

                    print("|File "*df*"| γt_cut = "*string(γᵢtₗᵢₘ)*"done.") 

                    # Convert cut ρ series:
                    ρ = convert(Vector{Matrix{ComplexF64}}, ρ)
                    t = convert(Vector{Float64}, t)

                    Δt = t[2]-t[1]
                    time_steps = length(t)

                    #print(" Assembling objective... ")

                    print("|"*string(γᵢ)*df*":obj Pade... ")
                    # Polynomial objectives with (1) Pade and (2) Simpsom methods
                    obj_pade += LiPoSID.pade_obj(ρ, t, H_symb, J_symb)
                    print("|"*string(γᵢ)*df*":obj Simpson... ")
                    obj_simp += LiPoSID.simpson_obj(ρ, t, H_symb, J_symb)
                    
                    print("|"*string(γᵢ)*df*":obj Kraus... ")
                    # Polynomial objective and constrain assuming (3) Kraus evolution
                    objk, constrk = LiPoSID.kraus_obj_constr(ρ, K_symb)
                    obj_kraus += objk 
                    constr_kraus += constrk #!!!!

                    println(string(γᵢ)*df*": obj done.|")
                   
                end # of case if γt cut is posible 

            end # of files (initial states) loop 

            if  obj_pade == obj_simp == obj_kraus == 0

                print("| !!! Warning the cut γt  = "*string(γᵢtₗᵢₘ)*" is not availbe.")
                println(" for γ = "*string(γᵢ)*" |")
    
            else 

                #println("Performing system identification with polynomial optimization ...")

                # Polynomial optimization assuming Lindblad evolution

                print("|"*string(γᵢ)*"POP Pade: ")
                solution_pade, tssos_iter_pade = LiPoSID.min2step(obj_pade)
                #println(" Number of TSSOS iterations: ", tssos_iter_pade)

                print("|"*string(γᵢ)*"POP Simpson: ")
                solution_simp, tssos_iter_simp = LiPoSID.min2step(obj_simp)
                #println(" Number of TSSOS iterations: ", tssos_iter_simp)

                H_sid_pade = subs(H_symb, solution_pade)
                J_sid_pade =  [ subs(J, solution_pade)  for J in J_symb ]
                
                H_sid_simp = subs(H_symb, solution_simp)
                J_sid_simp = [ subs(J, solution_simp) for J in J_symb ] 

                # Polynomial optimization assuming Kraus evolution
                print("|"*string(γᵢ)*"POP Kraus: ")
                solution_kraus, tssos_iter_kraus = LiPoSID.min2step(obj_kraus, constr_kraus)
                # println(" Number of TSSOS iterations: ", tssos_iter_kraus)
                K_sid = [ subs(K, solution_kraus)  for K in K_symb ]

                println(string(γᵢ)*"POP done.|")

                # Save results to HDF5

                print("| Saving for for γt cut ="*string(γᵢtₗᵢₘ))
                
                #lock(lck_write)
                    
                #try

                    h5open(directory*res_file_name,"cw") do fid  # read-write, create file if not existing, preserve existing contents

                        γt_cut_group = create_group(fid["gamma_"*string(γᵢ)], "gt_"*string(γᵢtₗᵢₘ))

                        γt_cut_group["H_sid_pade"] = convert.(ComplexF64, H_sid_pade)
                        γt_cut_group["J_sid_pade"] = reshape(reduce(hcat,[convert.(ComplexF64, J) for J in J_sid_pade]), 2, 2, :)
                        

                        γt_cut_group["H_sid_simp"] = convert.(ComplexF64, H_sid_simp)
                        γt_cut_group["J_sid_simp"] = reshape(reduce(hcat,[convert.(ComplexF64, J) for J in J_sid_simp]), 2, 2, :)
                    
                        γt_cut_group["K_sid"] = reshape(reduce(hcat,[convert.(ComplexF64, K) for K in K_sid]), 2, 2, :)
                        
                        γt_cut_group["tssos_iter_pade"] = tssos_iter_pade
                        γt_cut_group["tssos_iter_simp"] = tssos_iter_simp
                        γt_cut_group["tssos_iter_kraus"] = tssos_iter_kraus

                    end # of HDF5 writing

                #finally
                #    unlock(lck_write)
                #end

                println(" Saving for γt cut ="*string(γᵢtₗᵢₘ)*"done.|")

            end # of if case objectives not zero
        
        end # of loop over γt cuts
            
        println("|| Saving for γ ="*string(γᵢ)*" is done.||")
        
    end # of γ coupling (noise) levels loop

end # of operators num loop

# Identified operators are saved in HDF5 file for futher processing.

