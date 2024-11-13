abstract type MomentConditionsOrDerivatives end

macro gdg(args...)
    func = args[end]
    @capture(func, function (f_::s_)(xs__) body_ end) ||
        @capture(func, (f_::s_)(xs__) = body_) ||
        error("method definition is not recognized")
    return quote
        struct $(esc(s)){D} <: MomentConditionsOrDerivatives
            data::D
        end
        $(esc(func))
    end
end

show(io::IO, gdg::MomentConditionsOrDerivatives) = print(io, typeof(gdg).name.name)
show(io::IO, ::MIME"text/plain", gdg::MomentConditionsOrDerivatives) =
    print(io, typeof(gdg).name.name,
        " (functor defined with @gdg for moment conditions)")

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

function _parse_params(ps::Union{AbstractVector{<:Pair}, Base.Pairs})
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

function _parse_bayes_params(ps::Union{AbstractVector{<:Pair}, Base.Pairs})
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

function _parse_eqi(eq, nocons::Bool)
    if length(eq) == 3 # Manual specification of Z
        out = (eq[1], collect(VarName, eq[2]), collect(VarName, eq[3]))
    elseif length(eq) == 2
        xs = VarName[]
        zs = VarName[]
        for v in eq[2]
            if v isa Pair
                v[1] isa VarName ? push!(xs, v[1]) : append!(xs, v[1])
                v[2] isa VarName ? push!(zs, v[2]) : append!(zs, v[2])
            elseif v isa VarName
                push!(xs, v)
                push!(zs, v)
            else
                throw(ArgumentError("invalid specification of eqs"))
            end
        end
        out = (eq[1], xs, zs)
    else
        throw(ArgumentError("invalid specification of eqs"))
    end
    if !nocons
        :cons in out[2] || push!(out[2], :cons)
        :cons in out[3] || push!(out[3], :cons)
    end
    return out
end

# Only one equation entered without a vector
function _parse_eqs(eq, nocons::Bool)
    out = Vector{Tuple{VarName,Vector{VarName},Vector{VarName}}}(undef, 1)
    out[1] = _parse_eqi(eq, nocons)
    return out, copy(out[1][2])
end

function _parse_eqs(eqs::AbstractVector, nocons::Bool)
    out = Vector{Tuple{VarName,Vector{VarName},Vector{VarName}}}(undef, length(eqs))
    length(eqs) > 1 && (params = VarName[])
    for (i, eq) in enumerate(eqs)
        out[i] = _parse_eqi(eq, nocons)
        y = out[i][1]
        if length(eqs) > 1
            for n in out[i][2]
                push!(params, Symbol(y, "_", n))
            end
        else
            params = copy(out[1][2])
        end
    end
    return out, params
end

"""
    datafile(name::Union{Symbol,String})

Return the file path of the example data file named `name`.csv.gz.
"""
datafile(name::Union{Symbol,String}) = (@__DIR__)*"/../data/$(name).csv.gz"
