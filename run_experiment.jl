include("optimizer.jl")
using Main.Optimizer

experiment_name = ARGS[1]
job_id = ARGS[2]
experiment_id = ARGS[3]

# dataframe info
num_individuals = 48
f_ins = 0.7:0.01:1.0
p2s = [0.95, 0.99, 1.0]
num_simulations = 100000
save_to = experiment_name*"_"*job_id*"_"*experiment_id*".csv"

# values for initialization of population
n = 15
k = 11
r = 14
code_distance = 5
optimize_for = CostFunction(0) # 0: F_L, 1: F_A, 2: F_out
f_in = 0.9
p2 = 0.99
eta = 0.99
population_size = 200
starting_pop_multiplier = 20
max_gen = 250
max_ops = 50
starting_ops = 10
pairs = 20
children_per_pair = 3
mutants_per_individual_per_type = 5
p_single_operation_mutates = 0.1
p_lose_operation = 0.9
p_add_operation = 0.7
p_swap_operations = 0.8
p_mutate_operations = 0.8
individuals = []
selection_history = Dict()
num_simulations = 10000 


pop = Population(n, k, r, code_distance, optimize_for, f_in, p2, eta, population_size, starting_pop_multiplier, max_gen, max_ops, starting_ops, pairs, children_per_pair, mutants_per_individual_per_type, p_single_operation_mutates, p_lose_operation, p_add_operation, p_swap_operations, p_mutate_operations, individuals, selection_history, num_simulations)

run!(pop)
print(pop)

generate_dataframe(pop, num_individuals, f_ins, p2s, num_simulations, save_to)

