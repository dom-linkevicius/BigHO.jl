"""
    load_hyperoptimizer(objective, path) -> Hyperoptimizer

Reconstruct a `Hyperoptimizer` from a checkpoint written by `run!`'s
`save_path`, in exactly the state it was saved in -- runs, sampler state,
target, everything except the objective itself -- ready to keep going with
`run!(ho)` (raising the target with [`settarget!`](@ref) first if it had
already reached it). `objective` must be supplied fresh -- it's never part
of the checkpoint, since JLD2 can only serialize a function by name, which
isn't meaningful for an anonymous closure in a new session. Pass the same
objective you'd otherwise construct the `Hyperoptimizer` with.
"""
function load_hyperoptimizer(objective, path::AbstractString)
    saved = JLD2.jldopen(file -> file["ho"], path, "r")
    return Hyperoptimizer(saved.params, saved.candidates, saved.sampler, objective, saved.n,
                           saved.runs, saved.completed, saved.n_pending, saved.status,
                           saved.best_min_id, ReentrantLock())
end

# Writes ho to path atomically: objective replaced with `nothing` before
# serializing (never actually stored -- JLD2 can only serialize a function
# by name, meaningless for an anonymous closure in a future session) and
# lock replaced with a fresh one (a lock's internal state has no meaning
# across processes), written to a temp file in the same directory first,
# then renamed over the real path, so a crash mid-write can't corrupt or
# lose the last good checkpoint -- the rename either lands fully or not at
# all, never leaving path itself half-written. jldopen (not JLD2.save/load,
# which dispatch on the filename's extension via FileIO) is used
# throughout since the temp file's name doesn't end in .jld2.
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
