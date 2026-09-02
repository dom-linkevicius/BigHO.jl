"""
    load_hyperoptimizer(objective, path) -> Hyperoptimizer

Reconstruct a `Hyperoptimizer` from a `run!`'s `save_path` checkpoint, in exactly the state it was saved in.
`objective` must be supplied fresh -- JLD2 can't meaningfully serialize an anonymous closure, so it's never part of the checkpoint.
"""
function load_hyperoptimizer(objective, path::AbstractString)
    saved = JLD2.jldopen(file -> file["ho"], path, "r")
    return Hyperoptimizer(saved.params, saved.candidates, saved.sampler, objective, saved.n,
                           saved.runs, saved.completed, saved.n_pending, saved.status,
                           saved.best_min_id, ReentrantLock())
end

# objective/lock aren't meaningfully serializable, so both are replaced before writing.
# Atomic: written to a temp file first, then renamed over path, so a crash mid-write can't corrupt it.
function _save_checkpoint(ho::Hyperoptimizer, path::AbstractString)
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
