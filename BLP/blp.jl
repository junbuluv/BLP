using CSV, DataFrames

df = CSV.read("/Users/junbu/Documents/GitHub/BLP/BLP/data/data.csv", DataFrame)

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



#################### RANDOM COEFFICIENT LOGIT ####################













