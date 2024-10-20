macro gdg(args...)
    func = args[end]
    @capture(func, function (f_::s_)(xs__) body_ end) ||
        @capture(func, (f_::s_)(xs__) = body_) ||
        error("method definition is not recognized")
    return quote
        struct $(esc(s)){D}
            data::D
        end
        $(esc(func))
    end
end

function _default_ntasks(N::Integer)
    if N < 40_000
        return 1
    elseif N < 400_000
        return Threads.nthreads() ÷ 2
    else
        return Threads.nthreads()
    end
end

const VarName = Union{Symbol, Int}

function _parse_params(ps::Union{AbstractVector{<:Pair}, Dict, Base.Pairs})
    names = Vector{VarName}(undef, length(ps))
    initvals = Vector{Float64}(undef, length(ps))
    for (i, p) in enumerate(ps)
        names[i] = p[1]
        initvals[i] = p[2]
    end
    return names, initvals
end

_parse_params(ps::NamedTuple) = _parse_params(pairs(ps))

_parse_params(ps::AbstractVector{<:VarName}) =
    collect(VarName, ps), zeros(length(ps))

function _parse_params(ps::Tuple)
    names = Vector{VarName}(undef, length(ps))
    initvals = Vector{Float64}(undef, length(ps))
    for (i, p) in enumerate(ps)
        if p isa Pair
            names[i] = p[1]
            initvals[i] = p[2]
        elseif p isa VarName
            names[i] = p
            initvals[i] = 0.0
        else
            throw(ArgumentError("invalid specification of params"))
        end
    end
    return names, initvals
end

function _parse_bayes_params(ps::Union{AbstractVector{<:Pair}, Dict, Base.Pairs})
    names = Vector{VarName}(undef, length(ps))
    priors = Vector{Distribution}(undef, length(ps))
    for (i, p) in enumerate(ps)
        names[i] = p[1]
        priors[i] = p[2]
    end
    return names, (priors...,)
end

_parse_bayes_params(ps::NamedTuple) = _parse_bayes_params(pairs(ps))

function _parse_bayes_params(ps::Tuple)
    names = Vector{VarName}(undef, length(ps))
    priors = Vector{Distribution}(undef, length(ps))
    for (i, p) in enumerate(ps)
        if p isa Pair
            names[i] = p[1]
            priors[i] = p[2]
        else
            throw(ArgumentError("invalid specification of params"))
        end
    end
    return names, (priors...,)
end

function acceptance_rate(sample::AbstractVector)
    i1 = firstindex(sample)
    iN = lastindex(sample)
    count = 1
    vlast = sample[i1]
    @inbounds for i in i1+1:iN
        v = sample[i]
        if v != vlast
            count += 1
            vlast = v
        end
    end
    return count/(iN-i1+1)
end

"""
    datafile(name::Union{Symbol,String})

Return the file path of the example data file named `name`.csv.gz.
"""
datafile(name::Union{Symbol,String}) = (@__DIR__)*"/../data/$(name).csv.gz"
