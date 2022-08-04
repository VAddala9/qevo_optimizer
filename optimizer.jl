using QuantumClifford
using BPGates
using PyPlot
using QuantumClifford.Experimental.NoisyCircuits

struct Performance
    error_probabilities::Vector{Float64}
    success_probability::Float64
end

mutable struct Individual
    history::String
    k::UInt8
    r::UInt8
    f_in::Float64
    cost_function # function that takes in Performance type and returns a number
    ops::Vector{Any}
    performance::Performance
    fitness::Float64
end 

struct SimulationObject 
    successes::Vector{Bool} # each element corresponds to one monte carlo run
    bell_states::Vector{BellState}
end

p_zero(p::Performance)=p.error_probabilities[1]
p_one_zero(p::Performance)=sum(p.error_probabilities[1:2])
p_one_zero_two(p::Performance)=sum(p.error_probabilities[1:3])
    
function simulate_circuit(indiv::Individual, num_simulations)
    return SimulationObject([rand([true, false]) for i=1:num_simulations], [rand_state(indiv.r) for i=1:num_simulations])
end

function calculate_performance(indiv::Individual, num_simulations=10) # later pass in min_accuracy instead of num_simulations
    all_resulting_pairs = Vector{BellState}()
    all_successes = Vector{Bool}()
    for i=1:num_simulations
        resulting_pairs, success = mctrajectory!(BellState(zeros(Bool, 2*indiv.r)), indiv.ops)
        push!(all_resulting_pairs, resulting_pairs)
        push!(all_successes, success==QuantumClifford.Experimental.NoisyCircuits.CircuitStatus(0))
    end

    successful_runs = (all_successes .== 1)
    successes = sum(successful_runs)
    if successes == 0
        return 0.0
    end
    success_probability = successes/num_simulations
    errors = [sum(all_resulting_pairs[i].phases[1:2:2*indiv.k] .| all_resulting_pairs[i].phases[2:2:2*indiv.k]) for i=1:num_simulations if successful_runs[i]]
    error_probabilities = [sum(errors .== i)/successes for i=0:indiv.k] # TODO - split and then benchmark
    indiv.performance = Performance(error_probabilities, success_probability)
    indiv.fitness = indiv.cost_function(indiv.performance)
end

function random_bell_op(registers::UInt8)
    register1 = rand(1:registers)
    register2 = rand(1:registers)
    while register1 == register2
        register2 = rand(1:registers)
    end

    if rand() < 0.6
        return BellGateQC(rand(1:4), rand(1:20), rand(1:6), rand(1:6), register1, register2)
    end
    return BellMeasure(rand(1:3), rand(1:registers))
end

function drop_op(indiv::Individual) 
    new_indiv = deepcopy(indiv)
    deleteat!(new_indiv.ops, rand(1:length(new_indiv.ops)))
    new_indiv.history = "drop_m"
    return new_indiv
end

function gain_op(indiv::Individual)
    new_indiv = deepcopy(indiv)
    insert!(new_indiv.ops, rand(1:length(new_indiv.ops)), random_bell_op(new_indiv.r))
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

function mutate(gate::BellMeasure)
    return BellMeasure(rand(1:3), gate.sidx)
end

function mutate(gate::BellGateQC)
    return BellGateQC(rand(1:4), rand(1:20), rand(1:6), rand(1:6), gate.idx1, gate.idx2)
end

function mutate(indiv::Individual)
    new_indiv = deepcopy(indiv)
    new_indiv.ops = [mutate(gate) for gate in new_indiv.ops]
    new_indiv.history = "ops_m"
    return new_indiv
end

function new_child(indiv::Individual, indiv2::Individual)
    new_indiv = deepcopy(indiv)
    ops1, ops2 = indiv.ops, indiv2.ops
    if rand() < 0.5
        ops1 = ops1[end:-1:1]
    end

    if rand() < 0.5
        ops2 = ops2[end:-1:1]
    end
    new_indiv.ops = vcat(ops1[1:rand(1:length(ops1))], ops2[1:rand(1:length(ops2))])
    new_indiv.history = "child"
    return new_indiv
end

function total_raw_pairs(indiv::Individual) # priority 2

end

function generate_dataframe() # priority 3

end

mutable struct Population
    n::UInt8
    k::UInt8
    r::UInt8
    cost_function
    f_in::Float64
    p2::Float64
    η::Float64
    population_size::UInt32
    starting_pop_multiplier::UInt32
    max_gen::UInt32
    max_ops::UInt8
    starting_ops::UInt8
    pairs::UInt8
    children_per_pair::UInt8
    mutants_per_individual_per_type::UInt8
    p_single_operation_mutates::Float64
    p_lose_operation::Float64
    p_add_operation::Float64
    p_swap_operations::Float64
    p_mutate_operations::Float64
    individuals::Vector{Individual}
    selection_history::Dict{Tuple{Int64, String}, Int64}
    num_simulations::UInt32
end

function initialize_pop(population::Population)
    population.individuals = [Individual("random", population.k, population.r, population.f_in, population.cost_function, [random_bell_op(population.r) for j=1:population.starting_ops], Performance([], 0.0), 0.0) for i=1:population.population_size*population.starting_pop_multiplier]
end

function cull!(population::Population)
    population.individuals = population.individuals[1:population.population_size]
end

function sort!(population::Population) 
    Threads.@threads for indiv in population.individuals
        calculate_performance(indiv, population.num_simulations) 
    end
    population.individuals = sort(population.individuals, by = x -> x.fitness, rev=true)
end

function step(population::Population)
    for indiv in population.individuals
        indiv.history = "survivor"
    end

    parents = [(rand(population.individuals), rand(population.individuals)) for i=1:population.pairs]
    for (p1, p2) in parents
        population.individuals = vcat(population.individuals, [new_child(p1, p2) for j=1:population.children_per_pair])
    end

    for indiv in population.individuals[1:population.population_size]
        population.individuals = vcat(population.individuals, [drop_op(indiv) for i=1:population.mutants_per_individual_per_type if rand() < population.p_lose_operation && length(indiv.ops) > 0])
        population.individuals = vcat(population.individuals, [gain_op(indiv) for i=1:population.mutants_per_individual_per_type if rand() < population.p_add_operation && length(indiv.ops) < population.max_ops])
        population.individuals = vcat(population.individuals, [swap_op(indiv) for i=1:population.mutants_per_individual_per_type if rand() < population.p_swap_operations && length(indiv.ops) > 0])
        population.individuals = vcat(population.individuals, [mutate(indiv) for i=1:population.mutants_per_individual_per_type if rand() < population.p_mutate_operations && length(indiv.ops) > 0])
    end

    sort!(population)
    cull!(population)
end

function run(population::Population)
    hist_dict = Dict{Tuple{Int64, String}, Int64}()
    initialize_pop(population)
    sort!(population)
    cull!(population)
    for i=1:population.max_gen
        step(population)
        for hist in ["manual", "survivor", "random", "child", "drop_m", "gain_m", "swap_m", "ops_m"]
            hist_dict[(i, hist)] = reduce(+, [1 for indiv in population.individuals if indiv.history==hist], init=0)
            # hist_dict[hist] - array and then push on this
        end
    end
    population.selection_history = hist_dict
end

pop = Population(15, 11, 14, p_zero, 0.9, 0.99, 0.99, 200, 20, 100, 10, 6, 20, 3, 5, 0.1, 0.9, 0.7, 0.8, 0.8, [], Dict(), 10)
@benchmark run(pop)

