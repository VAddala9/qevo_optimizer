# n to k Optimized Entanglement Purification

`optimizer.jl` is a small Julia library for finding optimized n to k purification circuits. It makes use of a Monte Carlo simulator, which allows for a large speed up. Note that the calculated fidelities are not exact, and their error scales with $\frac{1}{\sqrt{N}}$ where $N$ is the number of simulations.

`examples/run_experiment.jl` contains an example run through of using the optimizer. An example output from an experiment run is located in `examples/data/example_output.csv`. `examples/plotting_example` runs through how some of the data can be visualized. `examples/submit.sh` is an example of what can be used for running simulations with a job array and multithreading on a supercomputing cluster.

`old_optimizer` contains the older python implementation, `qevo_ntom.py`, which calculates exact fidelities and has a few different features. The exact calculations come with an exponential increase in computation with the number of registers. `Example.ipynb` runs through an example purifying to two pairs.

### Future Improvements
- in calculate_performance!, first run the circuit with only initialization noise for fewer number of simulations and skip running the simulations with noisy operations if the fidelity is too low
- increasing the number of simulations as the generation increases
