"""
    NAGARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
"""
struct NAGARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function NAGARCH{p, q, T}(coefs::Vector{T}) where {p, q, T}
        length(coefs) == nparams(NAGARCH{p, q})  || throw(NumParamError(nparams(NAGARCH{p, q}), length(coefs)))
        new{p, q, T}(coefs)
    end
end





NAGARCH{p, q}(coefs::Vector{T}) where {p, q, T}  = NAGARCH{p, q, T}(coefs)

@inline nparams(::Type{<:NAGARCH{p, q}}) where {p, q} = p+2q+1
@inline nparams(::Type{<:NAGARCH{p, q}}, subset) where {p, q} = isempty(subset) ? 1 : sum(subset) + 1

@inline presample(::Type{<:NAGARCH{p, q}}) where {p, q} = max(p, q)






Base.@propagate_inbounds @inline function update!(
            ht, lht, zt, at, ::Type{<:NAGARCH{p, q}}, garchcoefs
            ) where {p, q}
	mht = garchcoefs[1]
    for i = 1:p
        mht += garchcoefs[i+1]*ht[end-i+1]
    end
    for i = 1:q
        mht += garchcoefs[i+1+p] * (at[end-i+1] + garchcoefs[i+1+p+q] * sqrt(ht[end-i+1]))^2
    end
    push!(ht, mht)
    push!(lht, (mht > 0) ? log(mht) : -mht)
    return nothing
end



# using ω / (1-α-αγ²-β)

@inline function uncond(::Type{<:NAGARCH{p, q}}, coefs::Vector{T}) where {p, q, T}
    den=one(T)
    for i = 1:p
        den -= coefs[i+1]
    end
    for i = 1:q
        den -= coefs[1+p+i]
    end
    for i = 1:q
        den -= coefs[1+p+i] * coefs[1+p+q+i]^2
    end
    h0 = coefs[1]/den
end



function startingvals(spec::Type{<:NAGARCH{p, q}}, data::Array{T}) where {p, q, T}
    x0 = zeros(T, p+2q+1)
    x0[1]=1
    x0[2:p+1] .= 0.9/p
    x0[p+2:end] .= 0.025/q
    x0[1] = var(data)/uncond(spec, x0)
    return x0
end

# using ω / (1-α-αγ²-β)

function startingvals(TT::Type{<:NAGARCH}, data::Array{T} , subset::Tuple) where {T}
	p, q = subsettuple(TT, subsetmask(TT, subset))
	x0 = zeros(T, p+2*q+1)
    x0[2:p+1] .= 0.9/p
    x0[p+2:end] .= 0.025/q
    x0[1] = var(data)*(one(T)-sum(x0[2:p+q+1])- sum(x0[p+q+2:end].^2 .* x0[p+2:p+q+1]))
	mask = subsetmask(TT, subset)
	x0long = zeros(T, length(mask))
	x0long[mask] .= x0
    return x0long
end

function constraints(::Type{<: NAGARCH{p,q}}, ::Type{T}) where {p, q, T}
    lower = zeros(T, p+2q+1)
    upper = ones(T, p+2q+1)
    upper[1] = T(Inf)
	return lower, upper
end


function coefnames(::Type{<:NAGARCH{p,q}}) where {p, q}
    names = Array{String, 1}(undef, p+2q+1)
    names[1] = "ω"
    names[2:p+1] .= (i -> "β"*subscript(i)).([1:p...])
    names[2+p:p+q+1] .= (i -> "α"*subscript(i)).([1:q...])
    names[p+q+2:p+2*q+1] .= (i -> "γ"*subscript(i)).([1:q...])
    return names
end

@inline function subsetmask(VS_large::Union{Type{NAGARCH{p, q}}, Type{NAGARCH{p, q, T}}}, subs) where {p, q, T}
	ind = falses(nparams(VS_large))
	subset = zeros(Int, 2)
	subset[3-length(subs):end] .= subs
	ind[1] = true
	ps = subset[1]
	qs = subset[2]
	@assert ps <= p
	@assert qs <= q
	ind[2:2+ps-1] .= true
	ind[2+p:2+p+qs-1] .= true
	ind[2+p+q:2+p+qs+q-1] .= true
	ind
end

@inline function subsettuple(VS_large::Union{Type{NAGARCH{p, q}}, Type{NAGARCH{p, q, T}}}, subsetmask) where {p, q, T}
	ps = 0
	qs = 0
	@inbounds @simd ivdep for i = 2 : p + 1
		ps += subsetmask[i]
	end
	@inbounds @simd ivdep for i = p + 2 : p + q + 1
		qs += subsetmask[i]
	end
	(ps, qs)
end
