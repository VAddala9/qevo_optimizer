This Python optimizer is almost identical to that in [qevo](https://github.com/Krastanov/qevo). The main modification is an added set of ways to optimize for purified bell pairs, which can be specified by ```WEIGHTS``` when initializing a ```Population``` object. The most useful weights are defined as follows:
- ```(1, 0)```: optimizing for the marginal fidelities of purified pairs (ignores correlated errors and is the default setting)
- ```Pzero```: optimizing to correct all errors (corresponds to purifiying for perfect teleportation)
- ```PzeroPone```: optimizing to correct errors on more than two qubits (can be useful when followed by an error correction protocol which will correct one-qubit errors)

The remaining weights are either slight variants of the above or were found to not work well for optimization. "I" corresponds to mutual information amongst the purified pairs and "S" corresponds to the average marginal entropy of the purified pairs.

