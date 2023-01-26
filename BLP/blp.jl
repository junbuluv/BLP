using CSV, DataFrames, ForwardDiff, NLsolve, LinearAlgebra, Optim, LineSearches, JuMP, Ipopt, Statistics, Random

df = CSV.read("C:/Users/16192/Documents/GitHub/BLP/BLP/data/data.csv", DataFrame)

# Table 3 column 1. Linear regression 
# Berry (1994) Inversion:
# log(s) - log(s_0) = X*β - α * P + ξ 

market_ids = unique(df.market_ids)
β = zeros(eltype(Float64), 6)
data = copy(df)
share_out = Vector{Float64}[]
t = first(market_ids)
for t in first(market_ids) : last(market_ids)
    s_0 = (1 -sum(data.shares[data.market_ids .== t,:]))
    new_share_out = ones(size(data.shares[data.market_ids .== t,:],1))*s_0
    share_out = vcat(share_out, new_share_out)
end

data.share_out = share_out
Y = log.(data.shares) - log.(data.share_out)
X = hcat(ones(size(data,1)), data.hpwt, data.air, data.mpd, data.space, data.prices)
β_ols = X \ Y


# Table 3 column 2. IV_logit demand 
# For IV, use demand_instrument 1 ~ 7
Z = hcat(ones(size(data,1)), data.hpwt, data.air, data.mpd, data.space, data.demand_instruments1, data.demand_instruments2, data.demand_instruments3, data.demand_instruments4, data.demand_instruments5,
data.demand_instruments6, data.demand_instruments7)
W = inv(Z'*Z)
β_IV = inv(X'*Z*W*Z'*X)*X'*Z*W*Z'*Y

########################################################################################################################

#################### RANDOM COEFFICIENT LOGIT ##########################################################################

########################################################################################################################
"""
Objective : Build MPEC approach BLP estimation
"""
mutable struct Marketdata
    """for market t in 1: T"""
    # marketshare : s_{1t} ~ s_{Jt} J X 1 vector
    s::AbstractVector

    # price : p_{1} ~ p_{J} : J X 1 vector
    p::AbstractVector

    # product characteristics : X_{11}~ X_{JK} (K X J) matrix
    x::AbstractMatrix

    # cost shifter : w_{11} ~ w_{JL} (L X J) matrix (cost shifters include some of the good characteristics)
    w::AbstractMatrix

    # firm_ids : there are M firms in market t
    firm_ids::AbstractVector

    # demandinstruments : zd_{11} ~ zd_{MJ} (M X J) matirx (Note: IV numbers must be bigger or equal to endogenous variable number)
    zd::AbstractMatrix

    # supplyinstruments : zs_{11} ~ zs_{HJ} (H X J) matrix (Note: IV numbers must be bigger or equal to endogenous variable number)
    zs::AbstractMatrix

    # monte_carlo integration : (K X S) matrix 
    ν::AbstractMatrix
end


"""
For market t : there are T markets
compute share : s_{jt} for j in 1:J

Input: price, good characteristics, simulated numbers , δ:: mean utility (x*β - α*p)
"""



function share(x::AbstractMatrix, σ::AbstractVector, δ::AbstractVector, ν::AbstractMatrix)
    # product number
    J = length(δ)
    # characteristics number
    K = size(σ,1)
    # simulation number
    S = size(ν,2)
    
    # initialize share : s_{1}~ s_{J} JX1 vector
    s = zeros(eltype(δ), size(δ))
    # monte_carlo integration part
    si = similar(s)
    σx = σ.*x'
    @inbounds for i in 1:S
        @simd   for j in 1:J
            @views si[j] = δ[j] + dot(σx[:,j], ν[:,i])
        end
        maxsi = max(maximum(si), 0)
        si .-= maxsi
        si  .= exp.(si)
        si .= si./(exp.(-maxsi) + sum(si))
        s .+= si
    end
    s ./= S
    s .+= eps(zero(eltype(s)))
    return s

end

"""
Delta contraction:
    After computing share, s_{1} ,...s_{J} 
    δ <- δ + log(S) - log(share_{1},...,{J}(β, σ))
    recover true δ

"""


function contraction(s::AbstractVector, δ::AbstractVector, σ::AbstractVector, x::AbstractMatrix, ν::AbstractMatrix)
    maxiter = 100000
    tol = 1e-14
    init = log.(s) .- log.(1-sum(s))
    sol = NLsolve.fixedpoint(delta->(delta .+ log.(s) .- log.(share(x, σ, delta, ν))), init , method=:anderson, show_trace=true)
    sol
    return sol.zero
end


"""
GMM part
"""








