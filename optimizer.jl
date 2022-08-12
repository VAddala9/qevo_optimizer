module Optimizer

using BPGates
using QuantumClifford
using QuantumClifford.Experimental.NoisyCircuits
using Random
#using PyPlot
using Statistics
using Revise

export Performance, Individual, p_zero, p_one_zero, p_two_one_zero,
    calculate_performance!, Population, initialize_pop!, cull!, sort!,
    step!, run!, fidelities, succ_probs

struct Performance
    error_probabilities::Vector{Float64}
    success_probability::Float64
end

mutable struct Individual
    history::String
    k::Int
    r::Int
    f_in::Float64
    cost_function # function that takes in Performance type and returns a number
    ops::Vector{Union{PauliNoiseBellGate{CNOTPerm},NoisyBellMeasureNoisyReset}}
    performance::Performance
    fitness::Float64
end 

p_zero(p::Performance)=p.error_probabilities[1]
p_one_zero(p::Performance)=sum(p.error_probabilities[1:2])
p_two_one_zero(p::Performance)=sum(p.error_probabilities[1:3])
    

function calculate_performance_new!(indiv::Individual, num_simulations=10) 
    K = indiv.k
    R = indiv.r
    state = BellState(R)
    count_success = 0
    counts_marginals = zeros(Int,K) # an array to find F₁, F₂, …, Fₖ
    counts_nb_errors = zeros(Int,K+1) # an array to find P₀, P₁, …, Pₖ -- Careful with indexing it!


    px0 = py0 = pz0 = (1-indiv.f_in)/3
    initial_noise_circuit = [PauliNoiseOp(i, px0, py0, pz0) for i in 1:indiv.r]
    
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
    indiv.performance = Performance(err_probs, p_success)
    indiv.fitness = count_success > 30 ? indiv.cost_function(indiv.performance) : 0.0
    indiv.performance
end


function calculate_performance!(indiv::Individual, num_simulations=10) 
    all_resulting_pairs = Vector{BellState}() # Todo - create in advance and assign to individual
    all_successes = Vector{Bool}()
    state = BellState(indiv.r)
    px0 = py0 = pz0 = (1-indiv.f_in)/3
    initial_noise_circuit = [PauliNoiseOp(i, px0, py0, pz0) for i in 1:indiv.r]
    for i=1:num_simulations
        new_state = copy(state)
        mctrajectory!(new_state, initial_noise_circuit)
        resulting_pairs, success = mctrajectory!(new_state, indiv.ops)
        push!(all_resulting_pairs, resulting_pairs)
        push!(all_successes, success==QuantumClifford.Experimental.NoisyCircuits.CircuitStatus(0))
    end

    successful_runs = (all_successes .== 1)
    successes = sum(successful_runs)
    if successes < 30
        indiv.performance = Performance(zeros(Float64, indiv.k), 0.0)
        return indiv.fitness = 0.0
    end
    success_probability = successes/num_simulations
    errors = [sum(all_resulting_pairs[i].phases[1:2:2*indiv.k] .| all_resulting_pairs[i].phases[2:2:2*indiv.k]) for i=1:num_simulations if successful_runs[i]]
    error_probabilities = [sum(errors .== i)/successes for i=0:indiv.k]
    indiv.performance = Performance(error_probabilities, success_probability)
    indiv.fitness = indiv.cost_function(indiv.performance)
    indiv.performance
end

function drop_op(indiv::Individual) 
    new_indiv = deepcopy(indiv)
    deleteat!(new_indiv.ops, rand(1:length(new_indiv.ops)))
    new_indiv.history = "drop_m"
    return new_indiv
end

function gain_op(indiv::Individual, p2::Float64, η::Float64)
    new_indiv = deepcopy(indiv)
    px = py = pz = (1 - p2)/4
    px0 = py0 = pz0 = (1-indiv.f_in)/3
    rand_op = rand() < 0.7 ? PauliNoiseBellGate(rand(CNOTPerm, randperm(indiv.r)[1:2]...), px, py, pz) : NoisyBellMeasureNoisyReset(rand(BellMeasure, rand(1:indiv.r)), 1-η, px0, py0, pz0)
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

function total_raw_pairs(indiv::Individual) # priority 2

end

function generate_dataframe() # priority 3

end

mutable struct Population
    n::Int
    k::Int
    r::Int
    cost_function
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

function initialize_pop!(population::Population)
    population.individuals = [Individual("random", population.k, population.r, population.f_in, population.cost_function, [], Performance([], 0.0), 0.0) for i=1:population.population_size*population.starting_pop_multiplier]
    Threads.@threads for indiv in population.individuals
        num_gates = rand(1:population.starting_ops-1)
        random_gates = [rand(CNOTPerm, (randperm(population.r)[1:2])...) for _ in 1:num_gates]
        px = py = pz = (1 - population.p2)/4
        px0 = py0 = pz0 = (1 - population.f_in)/3
        noisy_random_gates = [PauliNoiseBellGate(g, px, py, pz) for g in random_gates]
        random_measurements = [NoisyBellMeasureNoisyReset(rand(BellMeasure, rand(1:population.r)), 1-population.η, px0, py0, pz0) for _ in 1:(population.starting_ops-num_gates)]
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

