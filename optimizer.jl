module Optimizer

using BPGates
using QuantumClifford
using QuantumClifford.Experimental.NoisyCircuits
using Random
#using PyPlot
using Statistics
using Revise
using Pandas

export Performance, Individual, p_zero, p_one_zero, p_two_one_zero,
    calculate_performance!, Population, initialize_pop!, cull!, sort!,
    step!, run!, fidelities, succ_probs, renoise, total_raw_pairs, generate_dataframe # TODO - rethink this list

struct Performance
    error_probabilities::Vector{Float64}
    purified_pairs_fidelity::Float64
    logical_qubit_fidelity::Float64
    avg_marginals::Float64
    success_probability::Float64
end

mutable struct Individual
    history::String
    k::Int
    r::Int
    f_in::Float64
    code_distance::Int
    ops::Vector{Union{PauliNoiseBellGate{CNOTPerm},NoisyBellMeasureNoisyReset}}
    performance::Performance
    fitness::Float64
    optimize_marginals::Bool
end 

function Base.hash(indiv::Individual)
    return hash((Individual, indiv.k, indiv.r, [hash(op) for op in indiv.ops]))
end

function Base.hash(g::CNOTPerm)
    return hash((CNOTPerm, g.single1, g.single2, g.idx1, g.idx2))
end

function Base.hash(n::PauliNoiseBellGate{CNOTPerm})
    return hash((PauliNoiseBellGate{CNOTPerm}, n.px, n.py, n.pz))
end

function Base.hash(m::BellMeasure)
    return hash((BellMeasure, m.midx, m.sidx))
end

function Base.hash(n::NoisyBellMeasureNoisyReset)
    return hash((NoisyBellMeasureNoisyReset, hash(n.m), n.p, n.px, n.py, n.pz))
end

function calculate_performance!(indiv::Individual, num_simulations=10) 
    K = indiv.k
    R = indiv.r
    state = BellState(R)
    count_success = 0
    counts_marginals = zeros(Int,K) # an array to find F₁, F₂, …, Fₖ
    counts_nb_errors = zeros(Int,K+1) # an array to find P₀, P₁, …, Pₖ -- Careful with indexing it!


    initial_noise_circuit = [PauliNoiseOp(i, f_in_to_pauli(indiv.f_in)...) for i in 1:indiv.r]
    
    for _ in 1:num_simulations
        res_state, res = mctrajectory!(copy(state), initial_noise_circuit)
        res_state, res = mctrajectory!(res_state,indiv.ops)
        if res == continue_stat
            count_success += 1
            err_count = 0
            for i in 1:K
                if res_state.phases[2i-1] || res_state.phases[2i] # TODO write a better interface to get this data
                    err_count += 1
                else
                    counts_marginals[i] += 1
                end
            end
            counts_nb_errors[err_count+1] += 1
        end
    end

    p_success = count_success / num_simulations
    marginals = counts_marginals / count_success # it could have NaNs if count_success == 0
    err_probs = counts_nb_errors / count_success # it could have NaNs if count_success == 0
    correctable_errors = div(indiv.code_distance - 1, 2)
    logical_qubit_fidelity = sum(err_probs[1:min(end, correctable_errors+1)])
    indiv.performance = Performance(err_probs, err_probs[1], logical_qubit_fidelity, mean(marginals), p_success)

    indiv.fitness = indiv.optimize_marginals ? indiv.performance.avg_marginals : indiv.performance.logical_qubit_fidelity
    indiv.fitness = count_success > 0 ? indiv.fitness : 0.0
end

function drop_op(indiv::Individual) 
    new_indiv = deepcopy(indiv)
    deleteat!(new_indiv.ops, rand(1:length(new_indiv.ops)))
    new_indiv.history = "drop_m"
    return new_indiv
end

function gain_op(indiv::Individual, p2::Float64, η::Float64)
    new_indiv = deepcopy(indiv)
    rand_op = rand() < 0.7 ? PauliNoiseBellGate(rand(CNOTPerm, randperm(indiv.r)[1:2]...), p2_to_pauli(p2)...) : NoisyBellMeasureNoisyReset(rand(BellMeasure, rand(1:indiv.r)), 1-η, f_in_to_pauli(indiv.f_in)...)
    if length(new_indiv.ops) == 0
        push!(new_indiv.ops, rand_op)
    else
        insert!(new_indiv.ops, rand(1:length(new_indiv.ops)), rand_op)
    end

    new_indiv.history = "gain_m"
    return new_indiv
end

function swap_op(indiv::Individual)
    new_indiv = deepcopy(indiv)
    ind1, ind2 = rand(1:length(new_indiv.ops)), rand(1:length(new_indiv.ops))
    op1, op2 = new_indiv.ops[ind1], new_indiv.ops[ind2]
    new_indiv.ops[ind1] = op2
    new_indiv.ops[ind2] = op1
    new_indiv.history = "swap_m"
    return new_indiv
end

function mutate(gate::NoisyBellMeasureNoisyReset)
    return NoisyBellMeasureNoisyReset(rand(BellMeasure, gate.m.sidx), gate.p, gate.px, gate.py, gate.pz)
end

function mutate(gate::PauliNoiseBellGate)
    return PauliNoiseBellGate(rand(CNOTPerm, gate.g.idx1, gate.g.idx2), gate.px, gate.py, gate.pz)
end

function mutate(indiv::Individual)
    new_indiv = deepcopy(indiv)
    new_indiv.ops = [mutate(gate) for gate in new_indiv.ops]
    new_indiv.history = "ops_m"
    return new_indiv
end

function new_child(indiv::Individual, indiv2::Individual, max_ops::Int)
    new_indiv = deepcopy(indiv)
    ops1, ops2 = indiv.ops, indiv2.ops
    if rand() < 0.5
        ops1 = ops1[end:-1:1]
    end
    if length(ops1) == 0 || length(ops2) == 0
        return new_indiv
    end

    if rand() < 0.5
        ops2 = ops2[end:-1:1]
    end
    new_indiv.ops = vcat(ops1[1:rand(1:min(length(ops1), max_ops))], ops2[1:rand(1:length(ops2))])[1:min(end, max_ops)]
    new_indiv.history = "child"
    return new_indiv
end

function total_raw_pairs(indiv::Individual)
    total = indiv.r
    last_ops_reg = Set(1:indiv.r)
    for op in reverse(indiv.ops)
        if isa(op, NoisyBellMeasureNoisyReset)
            t = op.m.sidx
            if t in last_ops_reg
                delete!(last_ops_reg, t)
                if t < indiv.k
                    total+=1
                end
            else
                total+=1
            end
        else
            for t in [op.g.idx1, op.g.idx2]
                delete!(last_ops_reg, t)
            end
        end
    end
    return total
end

function f_in_to_pauli(f_in::Float64)
    px = py = pz = (1-f_in)/3
    return px, py, pz
end

function p2_to_pauli(p2::Float64)
    px = py = pz = (1-p2)/4
    return px, py, pz
end

function renoise(n::PauliNoiseBellGate{CNOTPerm}, f_in::Float64, p2::Float64)
    return PauliNoiseBellGate{CNOTPerm}(n.g, p2_to_pauli(p2)...)
end

function renoise(n::NoisyBellMeasureNoisyReset, f_in::Float64, p2::Float64)
    return NoisyBellMeasureNoisyReset(n.m, 1-p2, f_in_to_pauli(f_in)...)
end

function renoise(indiv::Individual, f_in::Float64, p2::Float64)
    return Individual(indiv.history, indiv.k, indiv.r, f_in, indiv.code_distance, [renoise(op, f_in, p2) for op in indiv.ops], Performance([], 0, 0, 0, 0), 0, indiv.optimize_marginals)
end

mutable struct Population
    n::Int
    k::Int
    r::Int
    code_distance::Int
    optimize_marginals::Bool
    f_in::Float64
    p2::Float64
    η::Float64
    population_size::Int
    starting_pop_multiplier::Int
    max_gen::Int
    max_ops::Int
    starting_ops::Int
    pairs::Int
    children_per_pair::Int
    mutants_per_individual_per_type::Int
    p_single_operation_mutates::Float64
    p_lose_operation::Float64
    p_add_operation::Float64
    p_swap_operations::Float64
    p_mutate_operations::Float64
    individuals::Vector{Individual}
    selection_history::Dict{String, Vector{Int64}}
    num_simulations::Int
end

function generate_dataframe(population::Population, f_ins, p2s, num_simulations, save_to)
    dataframe_length = length(population.individuals)*length(f_ins)*length(p2s)
    r = zeros(dataframe_length) # TODO - give all of these types
    k = zeros(dataframe_length)
    n = zeros(dataframe_length)
    circuit_length = zeros(dataframe_length)
    purified_pairs_fidelity = zeros(dataframe_length)
    logical_qubit_fidelity = zeros(dataframe_length)
    code_distance = zeros(dataframe_length)
    optimizing_marginals = zeros(dataframe_length)
    avg_marginals = zeros(dataframe_length)
    success_probability = zeros(dataframe_length)
    error_probabilities = repeat([[0.0]], dataframe_length)
    f_in = zeros(dataframe_length)
    p2 = zeros(dataframe_length)
    circuit_hash = zeros(dataframe_length)
    individual = repeat([""], dataframe_length)

    Threads.@threads for i1 in 1:length(population.individuals)
        indiv = population.individuals[i1]
        representative_indiv = renoise(indiv, 0.9, 0.99)
        indiv_repr = repr(representative_indiv)
        indiv_hash = hash(representative_indiv)
        for i2 in 1:length(f_ins)
            for i3 in 1:length(p2s)
                f = f_ins[i2]
                p = p2s[i3]
                index = (i1-1)*length(f_ins)*length(p2s) + (i2-1)*length(p2s) + (i3-1) + 1
                new_indiv = renoise(indiv, f, p)
                calculate_performance!(new_indiv, num_simulations)
                n[index] = total_raw_pairs(new_indiv)
                r[index] = new_indiv.r
                k[index] = new_indiv.k
                circuit_length[index] = length(new_indiv.ops)
                purified_pairs_fidelity[index] = new_indiv.performance.purified_pairs_fidelity
                logical_qubit_fidelity[index] = new_indiv.performance.logical_qubit_fidelity
                error_probabilities[index] = new_indiv.performance.error_probabilities
                avg_marginals[index] = new_indiv.performance.avg_marginals
                success_probability[index] = new_indiv.performance.success_probability
                f_in[index] = f
                p2[index] = p
                circuit_hash[index] = indiv_hash
                individual[index] = indiv_repr
            end
        end
    end
    df = DataFrame(Dict(:error_probabilities=>error_probabilities, :n=>n, :r=>r, :k=>k, :circuit_length=>circuit_length, :purified_pairs_fidelity=>purified_pairs_fidelity, :logical_qubit_fidelity=>logical_qubit_fidelity, :avg_marginals=>avg_marginals, :success_probability=>success_probability, :f_in=>f_in, :p2=>p2, :circuit_hash=>circuit_hash, :individual=>individual))
    to_csv(df, save_to)
end

function initialize_pop!(population::Population)
    population.individuals = [Individual("random", population.k, population.r, population.f_in, population.code_distance, [], Performance([], 0, 0, 0, 0), 0, population.optimize_marginals) for i=1:population.population_size*population.starting_pop_multiplier]
    Threads.@threads for indiv in population.individuals
        num_gates = rand(1:population.starting_ops-1)
        random_gates = [rand(CNOTPerm, (randperm(population.r)[1:2])...) for _ in 1:num_gates]
        noisy_random_gates = [PauliNoiseBellGate(g, p2_to_pauli(population.p2)...) for g in random_gates]
        random_measurements = [NoisyBellMeasureNoisyReset(rand(BellMeasure, rand(1:population.r)), 1-population.η, f_in_to_pauli(population.f_in)...) for _ in 1:(population.starting_ops-num_gates)]
        all_ops = vcat(noisy_random_gates, random_measurements)
        random_circuit = convert(Vector{Union{PauliNoiseBellGate{CNOTPerm},NoisyBellMeasureNoisyReset}}, all_ops[randperm(population.starting_ops)])
        indiv.ops = random_circuit
    end
end

function cull!(population::Population)
    population.individuals = population.individuals[1:population.population_size]
end

function sort!(population::Population) 
    Threads.@threads for indiv in population.individuals
        calculate_performance!(indiv, population.num_simulations) 
    end
    population.individuals = sort(population.individuals, by = x -> x.fitness, rev=true)
end

function step!(population::Population)
    for indiv in population.individuals
        indiv.history = "survivor"
    end

    parents = [(rand(population.individuals), rand(population.individuals)) for i=1:population.pairs]
    for (p1, p2) in parents
        population.individuals = vcat(population.individuals, [new_child(p1, p2, population.max_ops) for j=1:population.children_per_pair])
    end

    for indiv in population.individuals[1:population.population_size]
        population.individuals = vcat(population.individuals, [drop_op(indiv) for i=1:population.mutants_per_individual_per_type if rand() < population.p_lose_operation && length(indiv.ops) > 0])
        population.individuals = vcat(population.individuals, [gain_op(indiv, population.p2, population.η) for i=1:population.mutants_per_individual_per_type if rand() < population.p_add_operation && length(indiv.ops) < population.max_ops])
        population.individuals = vcat(population.individuals, [swap_op(indiv) for i=1:population.mutants_per_individual_per_type if rand() < population.p_swap_operations && length(indiv.ops) > 0])
        population.individuals = vcat(population.individuals, [mutate(indiv) for i=1:population.mutants_per_individual_per_type if rand() < population.p_mutate_operations && length(indiv.ops) > 0])
    end

    sort!(population)
    cull!(population)
end

function run!(population::Population)
    println(Threads.nthreads())
    for hist in ["manual", "survivor", "random", "child", "drop_m", "gain_m", "swap_m", "ops_m"]
        population.selection_history[hist] = Vector{Int64}()
    end
    initialize_pop!(population)
    sort!(population)
    cull!(population)
    for i=1:population.max_gen
        step!(population)
        for hist in ["manual", "survivor", "random", "child", "drop_m", "gain_m", "swap_m", "ops_m"]
            push!(population.selection_history[hist], reduce(+, [1 for indiv in population.individuals if indiv.history==hist], init=0))
        end
    end
end

function fidelities(population::Population)
    return [i.fitness for i in population.individuals]
end

function succ_probs(population::Population)
    return [i.performance.success_probability for i in population.individuals]
end

end # module

