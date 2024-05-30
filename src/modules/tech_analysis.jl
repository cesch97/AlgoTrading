
using DataFrames


function mean(x::AbstractArray)
    x = filter(!isnan, x)
    sum(x) / length(x)
end

function std(x::AbstractArray)
    x = filter(!isnan, x)
    sqrt(max(mean(x.^2) - mean(x)^2, 0.))
end

function calc_sma(data::DataFrame, period::Int)
    nan_vec = [NaN for i in 1:period-1]
    mean_vec = [mean(data.close[i-period+1: i]) for i in period:nrow(data)]
    [nan_vec; mean_vec]
end

function calc_sma(data::Vector, period::Int)
    nan_vec = [NaN for i in 1:period-1]
    mean_vec = [mean(data[i-period+1: i]) for i in period:length(data)]
    [nan_vec; mean_vec]
end

function calc_wma(data::DataFrame, period::Int)
    nan_vec = [NaN for i in 1:period-1]
    mean_vec = [sum(data.close[i-period+1: i] .* [1:period;]) / (period * (period+1) / 2) for i in period:nrow(data)]
    [nan_vec; mean_vec]
end

function calc_tr(data::DataFrame)
    h_l = data.high .- data.low
    h_c = abs.(data.high .- data.close)
    l_c = abs.(data.low .- data.close)
    max.(h_l, h_c, l_c)
end

function calc_atr(data::DataFrame, period::Int)
    tr = calc_tr(data)
    calc_sma(tr, period)
end

function calc_stoch_oscill(data::DataFrame, period::Int)
    nan_vec = [NaN for i in 1:period-1]
    close_prices = data.close[period:end]
    high_prices = [maximum(data.close[i-period+1: i]) for i in period:nrow(data)]
    low_prices = [minimum(data.close[i-period+1: i]) for i in period:nrow(data)]
    stoch_osc = clamp.((close_prices .- low_prices) ./ (high_prices .- low_prices) .* 100, 0, 100)
    [nan_vec; stoch_osc]
end

function calc_bb_up(data::DataFrame, period::Int, m::Number)
    nan_vec = [NaN for i in 1:period-1]
    tp = (data.high .+ data.low .+ data.close) ./ 3
    ind_vec = [mean(tp[i-period+1: i]) + (m * std(tp[i-period+1: i])) for i in period:nrow(data)]
    [nan_vec; ind_vec]
end

function calc_bb_down(data::DataFrame, period::Int, m::Number)
    nan_vec = [NaN for i in 1:period-1]
    tp = (data.high .+ data.low .+ data.close) ./ 3
    ind_vec = [mean(tp[i-period+1: i]) - (m * std(tp[i-period+1: i])) for i in period:nrow(data)]
    [nan_vec; ind_vec]
end


# indicators #
indicators_func = Dict(
    # prices
    "dist_close_open" => data::DataFrame -> data.close .- data.open,
    "dist_high_low" => data::DataFrame -> data.high .- data.low,
    "dist_high_open" => data::DataFrame -> data.high .- data.open,
    "dist_high_close" => data::DataFrame -> data.high .- data.close,
    "dist_open_low" => data::DataFrame -> data.open .- data.low,
    "dist_close_low" => data::DataFrame -> data.close .- data.low,
    # sma
    "dist_sma_open" => (data::DataFrame, period::Int) -> calc_sma(data, period) .- data.open,
    "dist_sma_high" => (data::DataFrame, period::Int) -> calc_sma(data, period) .- data.high,
    "dist_sma_low" => (data::DataFrame, period::Int) -> calc_sma(data, period) .- data.low,
    "dist_sma_close" => (data::DataFrame, period::Int) -> calc_sma(data, period) .- data.close,  
    "dist_sma_sma" => (data::DataFrame, period_1::Int, period_2::Int) -> calc_sma(data, period_1) .- calc_sma(data, period_2),
    # wma
    "dist_wma_open" => (data::DataFrame, period::Int) -> calc_wma(data, period) .- data.open,
    "dist_wma_high" => (data::DataFrame, period::Int) -> calc_wma(data, period) .- data.high,
    "dist_wma_low" => (data::DataFrame, period::Int) -> calc_wma(data, period) .- data.low,
    "dist_wma_close" => (data::DataFrame, period::Int) -> calc_wma(data, period) .- data.close,  
    "dist_wma_wma" => (data::DataFrame, period_1::Int, period_2::Int) -> calc_wma(data, period_1) .- calc_wma(data, period_2),
    "dist_wma_sma" => (data::DataFrame, period_1::Int, period_2::Int) -> calc_wma(data, period_1) .- calc_sma(data, period_2),
    # stoch_osc
    "stoch_osc" => (data::DataFrame, period::Int) -> calc_stoch_oscill(data, period),
    # atr
    "atr" => (data::DataFrame, period::Int) -> calc_atr(data, period),
    # BBs
    "dist_bbs" => (data::DataFrame, period::Int, m::Number) -> calc_bb_up(data, period, m) .- calc_bb_down(data, period, m),
    "dist_bb_up_open" => (data::DataFrame, period::Int, m::Number) -> calc_bb_up(data, period, m) .- data.open,
    "dist_bb_up_high" => (data::DataFrame, period::Int, m::Number) -> calc_bb_up(data, period, m) .- data.high,
    "dist_bb_up_low" => (data::DataFrame, period::Int, m::Number) -> calc_bb_up(data, period, m) .- data.low,
    "dist_bb_up_close" => (data::DataFrame, period::Int, m::Number) -> calc_bb_up(data, period, m) .- data.close,
    "dist_open_bb_down" => (data::DataFrame, period::Int, m::Number) -> data.open .- calc_bb_down(data, period, m),
    "dist_high_bb_down" => (data::DataFrame, period::Int, m::Number) -> data.high .- calc_bb_down(data, period, m),
    "dist_low_bb_down" => (data::DataFrame, period::Int, m::Number) -> data.low .- calc_bb_down(data, period, m),
    "dist_close_bb_down" => (data::DataFrame, period::Int, m::Number) -> data.close .- calc_bb_down(data, period, m),
    "dist_bb_up_sma" => (data::DataFrame, period_bb::Int, m::Number, period_sma::Int) -> calc_bb_up(data, period_bb, m) .- calc_sma(data, period_sma),
    "dist_sma_bb_down" => (data::DataFrame, period_bb::Int, m::Number, period_sma::Int) -> calc_sma(data, period_sma) .- calc_bb_down(data, period_bb, m),
    "dist_bb_up_wma" => (data::DataFrame, period_bb::Int, m::Number, period_wma::Int) -> calc_bb_up(data, period_bb, m) .- calc_wma(data, period_wma),
    "dist_wma_bb_down" => (data::DataFrame, period_bb::Int, m::Number, period_wma::Int) -> calc_wma(data, period_wma) .- calc_bb_down(data, period_bb, m),
)

indicators_params = Dict(
    # prices
    "dist_close_open" => (),
    "dist_high_low" => (),
    "dist_high_open" => (),
    "dist_high_close" => (),
    "dist_open_low" => (),
    "dist_close_low" => (),
    # sma
    "dist_sma_open" => (3:100, ),
    "dist_sma_high" => (3:100, ),
    "dist_sma_low" => (3:100, ),
    "dist_sma_close" => (3:100, ),  
    "dist_sma_sma" => (3:100, 3:100, ),
    # wma
    "dist_wma_open" => (3:100, ),
    "dist_wma_high" => (3:100, ),
    "dist_wma_low" => (3:100, ),
    "dist_wma_close" => (3:100, ),  
    "dist_wma_sma" => (3:100, 3:100, ),
    "dist_wma_wma" => (3:100, 3:100, ),
    #stoch_osc
    "stoch_osc" => (3:100, ),
    # atr
    "atr" => (3:100, ),
    # BBs
    "dist_bbs" => (3:100, 0.1:0.1:4, ),
    "dist_bb_up_open" => (3:100, 0.1:0.1:4, ),
    "dist_bb_up_high" => (3:100, 0.1:0.1:4, ),
    "dist_bb_up_low" => (3:100, 0.1:0.1:4, ),
    "dist_bb_up_close" => (3:100, 0.1:0.1:4, ),
    "dist_open_bb_down" => (3:100, 0.1:0.1:4, ),
    "dist_high_bb_down" => (3:100, 0.1:0.1:4, ),
    "dist_low_bb_down" => (3:100, 0.1:0.1:4, ),
    "dist_close_bb_down" => (3:100, 0.1:0.1:4, ),
    "dist_bb_up_sma" => (3:100, 0.1:0.1:4, 3:100, ),
    "dist_sma_bb_down" => (3:100, 0.1:0.1:4, 3:100, ),
    "dist_bb_up_wma" => (3:100, 0.1:0.1:4, 3:100, ),
    "dist_wma_bb_down" => (3:100, 0.1:0.1:4, 3:100, ),
)

# Selectors #
selectors_func = Dict(
    # last values
    "last_value" => indicator::Vector -> indicator,
    "mean_last_n_values" => (indicator::Vector, period::Int) -> calc_sma(indicator, period),
    "max_last_n_values" => (indicator::Vector, period::Int) -> [[NaN for i in 1:period-1]; [maximum(indicator[i-period+1: i]) for i in period:length(indicator)]],
    "min_last_n_values" => (indicator::Vector, period::Int) -> [[NaN for i in 1:period-1]; [minimum(indicator[i-period+1: i]) for i in period:length(indicator)]],
    # range
    "last_i_value" => (indicator::Vector, index::Int) -> [[NaN for i in 1:index-1]; [indicator[i-index+1] for i in index:length(indicator)]],
    "mean_range" => (indicator::Vector, start::Int, len::Float64) -> [[NaN for i in 1:start-1]; [mean(indicator[i-start+1: max(i - floor(Int, (1 - len) * start), i-start+1)]) for i in start:length(indicator)]],
    "max_range" => (indicator::Vector, start::Int, len::Float64) -> [[NaN for i in 1:start-1]; [maximum(indicator[i-start+1: max(i - floor(Int, (1 - len) * start), i-start+1)]) for i in start:length(indicator)]],
    "min_range" => (indicator::Vector, start::Int, len::Float64) -> [[NaN for i in 1:start-1]; [minimum(indicator[i-start+1: max(i - floor(Int, (1 - len) * start), i-start+1)]) for i in start:length(indicator)]],
)

selectors_params = Dict(
    # last values
    "last_value" => (),
    "mean_last_n_values" => (3:100, ),
    "max_last_n_values" => (3:100, ),
    "min_last_n_values" => (3:100, ),
    # range
    "last_i_value" => (3:100, ),
    "mean_range" => (3:100, 0.01:0.01:0.99, ),
    "max_range" => (3:100, 0.01:0.01:0.99, ),
    "min_range" => (3:100, 0.01:0.01:0.99, ),

)