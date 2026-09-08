# BigHO

## Introduction

BigHO.jl is a hyperparameter optimization library meant to be used in situation where individual function evaluations are costly. The package implements executors (which control how individual functions are evaluated) and samplers (how are the hyperparameters sampled), allowing for different composition of executors and samplers, depending on the optimization problem and available resources. The package is still under development and some of the features or implementations may change.

### Executors
The package implements three different executors, meant for different scenarios:
- `Serial`, which runs the sampler sequentially, mostly for development and testing purposes
- `Threaded`, which runs the sampler over the specified number of threads, meant for intermediate sized jobs
- `DistributedQueue` meant for the biggest jobs, usually on a compute cluster, running the sampler over different processes (one function evaluation per spawned process), requiring the specification of how those processes should be created (e.g. each process should have 12 threads).

### Samplers
The package currently implements the following samplers:
- `RandomSampler`, which selects the parameters randomly
- `LHSampler`, which is a Latin Hypercube sampler, aiming to maximally spread out `n` samples in the parameter space
- `Hyperband`, which runs brackets of trials, using a different number of resources to train them and promoting best trials within a bracket, until the bracket reaches it's resource limit
- `ASHA`, which is the asynchronous version of Hyperband, also running brackets, but not requiring the completion of a bracket before trials are promoted

## Convenience functionality

BigHO.jl provides some convenience functionality, such as
- tracking the status of individual trials, such that individual trial failures would not crash the whole optimization run, but should provide enough information to be reproducible
- saving results after each `k` trials in a user specified directory, such that after unexpected crashes and failures it would be possible to easily resume an optimization run
- conversion of a hyperparameter optimization results into a `DataFrames.DataFrame` for easier downstream analysis
- summary plotting (via a `CairoMakie` extension), showing scatter plots of the objective value against each hyperparameter (with marginal histograms), the objective value over trial id, and the best value found so far over trial id
- `Stateful` optimization functions, which allow continuation of training in samplers such as `Hyperband` or `ASHA`, amortizing some of the optimization costs

## Internal benchmarking


## Provenance

BigHO.jl started as a fork of, and was inspired by, [Hyperopt.jl](https://github.com/baggepinnen/Hyperopt.jl), but has since been rewritten essentially from the ground up to address some of the perceived limitations of that package.
