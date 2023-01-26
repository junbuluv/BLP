

using DataFrames, CSV, ForwardDiff, NLsolve, LinearAlgebra, Optim, LineSearches, JuMP, Ipopt, Statistics, Random






"""
Objective : Build MPEC approach BLP estimation
"""
mutable struct Marketdata
    """for market t in 1: T"""
    # marketshare : s_{1t} ~ s_{Jt}
    s::AbstractVector

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


#import data 
function import_data()
    csvfile = normpath("C:/Users/16192/JuliaProjects/BLP/data/blp_1999_data.csv")
    df = CSV.read(csvfile, DataFrame)
end

df = import_data()
# Clean data
data = select!(df,Not(:Column19))


"""
For market t : there are T markets
compute share : s_{jt} for j in 1:J

Input: price, good characteristics, simulated numbers , δ
"""


function share(x::AbstractMatrix, σ::AbstractVector, δ::AbstractVector, ν::AbstractMatrix)
    # product number
    J = length(δ)
    # characteristics number
    K = length(σ)
    # simulation number
    S = size(ν,2)
    
    # initialize share : s_{1}~ s_{J} JX1 vector
    s = zeros(eltype(δ), size(δ))
    # monte_carlo integration part
    si = similar(s)
    σx = σ.*x'
    @inbounds for i in 1:S
        @simd   for j in 1:J
            @views si[j] = δ[j] + dot(σx[:,j], ν[i,:])
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
    tol = 1e-5
    init = log.(s) .- log.(1-sum(s))
    sol = NLsolve.fixedpoint(delta->(delta .+ log.(s) .- log.(share(x, σ, delta, ν))), init)
    sol
    return sol.zero
end



"""
GMM part
"""



