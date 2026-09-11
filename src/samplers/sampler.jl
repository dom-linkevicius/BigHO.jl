"""
    Sampler
"""
abstract type Sampler end

"""
    on_tell!(sampler, runs, entry)
"""
function on_tell! end

"""
    init(sampler, candidates, n) -> sampler
"""
function init end

"""
    exhausted(sampler, ho) -> Bool
"""
function exhausted end

"""
    blocked(sampler, ho) -> Bool
"""
function blocked end

"""
    create_run_entry(sampler, ho, id, params, unit_params) -> RunEntry
"""
function create_run_entry end
