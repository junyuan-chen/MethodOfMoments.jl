struct RobustVCE{TF<:AbstractFloat} <: CovarianceEstimator
    S::Matrix{TF}
    dofr::Int
    vcovcache1::Matrix{TF}
    vcovcache2::Matrix{TF}
end

function RobustVCE(nparam::Integer, nmoment::Integer, nobs::Integer;
        adjustdofr::Integer=0, TF::Type=Float64)
    S = Matrix{TF}(undef, nmoment, nmoment)
    ca1 = similar(S, nmoment, nparam)
    ca2 = similar(S, nparam, nmoment)
    return RobustVCE(S, convert(Int,nobs-adjustdofr), ca1, ca2)
end

function setS!(vce::RobustVCE, res::AbstractMatrix)
    mul!(vce.S, res, res')
    vce.S ./= vce.dofr
    return vce.S
end

Base.show(io::IO, ::RobustVCE) =
    print(io, "Heteroskedasticity-robust covariance estimator")

struct ClusterVCE{TF<:AbstractFloat,TG} <: CovarianceEstimator
    clusternames::Vector{VarName}
    clusters::Vector{GroupedVector{TG}}
    G::Int
    us::Vector{Matrix{TF}}
    S::Matrix{TF}
    Sadj::TF
    vcovcache1::Matrix{TF}
    vcovcache2::Matrix{TF}
end

function ClusterVCE(data, clusternames, nparam::Integer, nmoment::Integer;
        Sadj::Union{Real,Nothing}=nothing, TF::Type=Float64)
    Tables.istable(data) ||
        throw(ArgumentError("data must be a Tables.jl-compatible table"))
    Nobs = length(Tables.rows(data))
    clusternames isa Symbol && (clusternames = (clusternames,))
    clusternames = VarName[clusternames...]
    nclu = length(clusternames)
    clusters = map(n->GroupedArray(Tables.getcolumn(data, n), sort=nothing), clusternames)
    G = minimum(x->x.ngroups, clusters)
    for n in 2:nclu
        for c in combinations(1:nclu, n)
            push!(clusters, GroupedArray((clusters[i] for i in c)..., sort=nothing))
        end
    end
    S = Matrix{TF}(undef, nmoment, nmoment)
    us = Vector{Matrix{TF}}(undef, length(clusters))
    for (i, g) in enumerate(clusters)
        us[i] = Matrix{TF}(undef, g.ngroups, nmoment)
    end
    Sadj === nothing && (Sadj = 1/Nobs)
    ca1 = similar(S, nmoment, nparam)
    ca2 = similar(S, nparam, nmoment)
    return ClusterVCE(clusternames, clusters, G, us, S, convert(TF,Sadj), ca1, ca2)
end

function setS!(vce::ClusterVCE{TF}, res::AbstractMatrix) where TF
    nclu = length(vce.clusternames)
    fill!(vce.S, zero(TF))
    k = 0
    for n in 1:nclu
        for c in combinations(1:nclu, n)
            k += 1
            u = vce.us[k]
            fill!(u, zero(TF))
            g = vce.clusters[k]
            # Residual matrix is assumed to be horizontal
            for i in axes(res,2)
                c = g.groups[i]
                @inbounds for j in axes(res,1)
                    u[c,j] += res[j,i]
                end
            end
            mul!(vce.S, u', u, vce.Sadj * (-1)^(length(c) - 1), 1)
        end
    end
    return vce.S
end

Base.show(io::IO, v::ClusterVCE) =
    print(io, "Cluster-robust covariance estimator: ", join(v.clusternames, ", "))

function Base.show(io::IO, ::MIME"text/plain", v::ClusterVCE)
    println(io, length(v.clusternames), "-way cluster-robust covariance estimator:")
    print(io, "  ", join(v.clusternames, ", "))
end
