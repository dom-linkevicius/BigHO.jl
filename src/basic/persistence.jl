"""
    load_hyperoptimizer(objective, path) -> Hyperoptimizer

Reconstruct a `Hyperoptimizer` from a checkpoint written by [`save_hyperoptimizer`](@ref) or `run!`'s `save_path`, in exactly the state it was saved in.
`objective` must be supplied fresh -- JLD2 can't meaningfully serialize an anonymous closure, so it's never part of the checkpoint.
"""
function load_hyperoptimizer(objective, path::AbstractString)
    saved = JLD2.jldopen(file -> file["ho"], path, "r")
    return Hyperoptimizer(saved.params, saved.candidates, saved.sampler, objective, saved.n,
                           saved.runs, saved.completed, saved.n_pending, saved.status,
                           saved.best_min_id, ReentrantLock())
end

"""
    save_hyperoptimizer(ho, path)

Write `ho` to `path` as a checkpoint, for reloading with [`load_hyperoptimizer`](@ref) -- the
same thing `run!`'s `save_path` writes, callable directly to save once outside a `run!`.
The objective and lock aren't meaningfully serializable, so both are replaced before writing.
Atomic: written to a temp file first, then renamed over `path`, so a crash mid-write can't corrupt it.
"""
function save_hyperoptimizer(ho::Hyperoptimizer, path::AbstractString)
    sanitized = Hyperoptimizer(ho.params, ho.candidates, ho.sampler, nothing, ho.n,
                                ho.runs, ho.completed, ho.n_pending, ho.status,
                                ho.best_min_id, ReentrantLock())
    tmp_path = path * ".tmp"
    JLD2.jldopen(tmp_path, "w") do file
        file["ho"] = sanitized
    end
    mv(tmp_path, path; force=true)
    return nothing
end
