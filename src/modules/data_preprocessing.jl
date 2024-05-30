
###########################
#                         #
#   Data Preprocessing    #
#                         #
###########################

"""
Te module provide methods and structurs to handle all the data used in the project
It relies on the module "tech_analysys for calculating indicators and selectors and it's used
by almost all the other modules
"""


using CSV
using Plots
using DataStructures
using DataFrames
using Dates
using Query
include("tech_analysis.jl")


### Structures ####

mutable struct StrategyData
    indicators::Array{NamedTuple,1}
    selectors::Array{NamedTuple,1}
    std_params::OrderedDict
end

mutable struct TradData
    symbol
    time_frame::String
    raw_data
    ind_data::OrderedDict
    sel_data::OrderedDict
    std_data::OrderedDict
end

function slice_by_time(data::DataFrame, date_from=nothing, date_to=nothing)::DataFrame
    if !(date_from === nothing) && !(date_to === nothing)
        data = filter(x -> (x.datetime >= date_from) && (x.datetime <= date_to), data)
    elseif !(date_from === nothing)
        data = filter(x -> x.datetime >= date_from, data)
    elseif !(date_to === nothing)
        data = filter(x -> x.datetime <= date_to, data)
    end
    data
end

function move_index(data::DataFrame)::DataFrame
    datetime = data.datetime[2:end]
    price = data.open[2:end]
    open = data.open[1:end-1]
    high = data.high[1:end-1]
    low = data.low[1:end-1]
    close = data.close[1:end-1]
    DataFrame(datetime = datetime, price = price, open = open, high = high, low = low, close = close)
end

function load_csv_file(data_dir::String, symbol::String, time_frame::String, date_from=nothing, date_to=nothing)
    file_dir = "$data_dir/$symbol/$time_frame/"
    all_files = readdir(file_dir, join=false, sort=true)
    data_years = parse.([Int,], replace.(all_files, [".csv" => ""]))
    if !isnothing(date_from)
        data_years = data_years[data_years .>= year(date_from)]
    end
    if !isnothing(date_to)
        data_years = data_years[data_years .<= year(date_to)]
    end
    for i in eachindex(data_years)
        year = data_years[i]
        header = ["datetime", "time", "open", "high", "low", "close", "volume"]
        data_chunk = CSV.File(file_dir * "/" * string(year) * ".csv", header=header) |> DataFrame
        data_chunk.datetime = Date.(data_chunk.datetime, DateFormat("y.m.d")) + data_chunk.time
        data_chunk = select(data_chunk, Not([:time, :volume]))
        if i == 1
            raw_data = data_chunk
        else
            raw_data = vcat(raw_data, data_chunk)
        end
    end
    raw_data = slice_by_time(raw_data, date_from, date_to)
    raw_data = move_index(raw_data)
    TradData(symbol, time_frame, raw_data, OrderedDict(), OrderedDict(), OrderedDict())
end

function standardize(x::Vector)
    μ = mean(x)
    σ = max(std(x), 1e-6)
    ((x .- μ) ./ σ), μ, σ
end

function un_standardize(x::Vector, μ::Number, σ::Number)
    x .* σ .+ μ
end

function calc_indicator(ind_name::String, ind_params, data::DataFrame)
    ind_func = indicators_func[ind_name]
    ind_func(data, ind_params...)    
end

function calc_selector(sel_name::String, sel_params, indicator::Vector)
    sel_func = selectors_func[sel_name]
    sel_func(indicator, sel_params...)    
end

function get_ind_id(ind::NamedTuple)
    id = ind.name
    for p in ind.params
        id *= "-" * string(p)
        if :symbol in keys(ind)
            id *= "-" * ind.symbol
        end
    end
    id
end

function get_sel_id(name::String, params, indicator)
    id = name
    for p in params
        id *= "-" * string(p)
    end
    id * "_" * string(indicator)
end

function clean_id(id::String)
    if occursin("#", id)
        return split(id, "#")[1]
    else
        return id
    end
end

function update_trad_data!(trad_data::TradData, strat_data::StrategyData; force_std::Bool=true)

    if isa(trad_data.symbol, Array{String,1})
        multi_symbol = true
    else
        multi_symbol = false
    end

    used_indicators = Int[] # avoid calculation of unused indicators
    for sel in strat_data.selectors
        ind = sel.indicator
        if !(ind in used_indicators)
            push!(used_indicators, ind)
        end
    end

    ind_data = OrderedDict()
    j = 0
    for (i, ind) in enumerate(strat_data.indicators)
        ind_id = get_ind_id(ind)
        if !haskey(ind_data, ind_id)
            if i in used_indicators
                if haskey(trad_data.ind_data, ind_id)
                    ind_data[ind_id] = trad_data.ind_data[ind_id]
                else
                    if !multi_symbol
                        ind_data[ind_id] = calc_indicator(ind.name, ind.params, trad_data.raw_data)
                    else
                        ind_data[ind_id] = calc_indicator(ind.name, ind.params, trad_data.raw_data[ind.symbol])
                    end
                end
            else
                ind_data[string(i)] = NaN # placeholder for non-calculated indicators
            end
        else
            j += 1
            ind_data[ind_id * "#" * string(j)] = ind_data[ind_id]
        end
    end
    sel_data = OrderedDict()
    std_data = OrderedDict()
    std_params = OrderedDict()
    i = 0
    for sel in strat_data.selectors
        sel_id = get_sel_id(sel.name, sel.params, sel.indicator)
        if !haskey(sel_data, sel_id)
            ind_id = clean_id(collect(keys(ind_data))[sel.indicator])
            if haskey(trad_data.sel_data, sel_id)
                old_ind_id = clean_id(collect(keys(trad_data.ind_data))[sel.indicator])
                if old_ind_id == ind_id
                    sel_data[sel_id] = trad_data.sel_data[sel_id]
                    std_data[sel_id] = trad_data.std_data[sel_id]
                    if haskey(strat_data.std_params, sel_id)
                        std_params[sel_id] = strat_data.std_params[sel_id]    
                    else
                        _, μ, σ = standardize(sel_data[sel_id])
                        std_params[sel_id] = (μ, σ) # (μ=μ, σ=σ)
                    end
                else
                    sel_data[sel_id] = calc_selector(sel.name, sel.params, ind_data[ind_id])
                    if force_std
                        std_vec, μ, σ = standardize(sel_data[sel_id])
                    else
                        μ, σ = strat_data.std_params[sel_id]
                        std_vec = (sel_data[sel_id] .- μ) ./ σ
                    end
                    std_data[sel_id] = std_vec
                    std_params[sel_id] = (μ, σ) # (μ=μ, σ=σ)                    
                end
            else
                sel_data[sel_id] = calc_selector(sel.name, sel.params, ind_data[ind_id])
                if force_std
                    std_vec, μ, σ = standardize(sel_data[sel_id])
                else
                    μ, σ = strat_data.std_params[sel_id]
                    std_vec = (sel_data[sel_id] .- μ) ./ σ
                end
                std_data[sel_id] = std_vec
                std_params[sel_id] = (μ, σ) # (μ=μ, σ=σ)
            end
        else
            i += 1
            sel_data[sel_id * "#" * string(i)] = sel_data[sel_id]
            std_data[sel_id * "#" * string(i)] = std_data[sel_id]
            std_params[sel_id * "#" * string(i)] = std_params[sel_id]
        end
    end
    strat_data.std_params = std_params
    trad_data.ind_data = ind_data
    trad_data.sel_data = sel_data
    trad_data.std_data = std_data
end

function make_sim_data(trad_data::TradData, date_from=nothing, date_to=nothing)
    if isa(trad_data.raw_data, DataFrame)
        df = DataFrame(trad_data.std_data)
        df.datetime = trad_data.raw_data.datetime
        df.price = trad_data.raw_data.price
        df.open = trad_data.raw_data.open
        df.high = trad_data.raw_data.high
        df.low = trad_data.raw_data.low
        df.close = trad_data.raw_data.close
        df = slice_by_time(df, date_from, date_to)
        mat = Matrix{Float32}(select(df, Not([:datetime, :price, :open, :high, :low, :close])))
        price_data = select(df, [:datetime, :price, :open, :high, :low, :close])
        return price_data, mat
    else
        df = DataFrame()
        std_data = DataFrame(trad_data.std_data)
        std_data.datetime = trad_data.raw_data[collect(keys(trad_data.raw_data))[1]][!, :datetime]
        last_datetime = std_data[1, :datetime]
        if dayofweek(last_datetime) == 7
            last_week = week(last_datetime) + 1
        else
            last_week = week(last_datetime)
        end
        for i in 2:nrow(std_data)
            datetime = std_data[i, :datetime]
            if (week(datetime) != last_week) && (day(datetime) != day(last_datetime))
                push!(df, std_data[i, :])
                last_datetime = std_data[i, :datetime]
                last_week = week(last_datetime)
            elseif (dayofweek(datetime) == 7) && (day(datetime) != day(last_datetime))
                push!(df, std_data[i, :])
                last_datetime = std_data[i, :datetime]
                last_week = week(last_datetime) + 1
            end
        end       
        df = slice_by_time(df, date_from-Day(1), date_to)      
        return df.datetime, Matrix{Float32}(select(df, Not(:datetime)))
    end
end

function get_rand_indicator()
    ind_name = rand(collect(keys(indicators_func)))
    ind_params = Number[]
    for param_val in indicators_params[ind_name]
        if isa(param_val, AbstractRange) || isa(param_val, Array)
            push!(ind_params, rand(param_val))
        else
            push!(ind_params, param_val)
        end
    end
    ind_params = tuple(ind_params...)
    (name=ind_name, params=ind_params)
end

function get_rand_indicator(symbols::Array{String,1})
    ind_name = rand(collect(keys(indicators_func)))
    ind_params = Number[]
    for param_val in indicators_params[ind_name]
        if isa(param_val, AbstractRange) || isa(param_val, Array)
            push!(ind_params, rand(param_val))
        else
            push!(ind_params, param_val)
        end
    end
    ind_params = tuple(ind_params...)
    (name=ind_name, params=ind_params, symbol=rand(symbols))
end

function get_rand_selector(num_indicators::Int)
    sel_name = rand(collect(keys(selectors_func)))
    sel_params = Number[]
    for param_val in selectors_params[sel_name]
        if isa(param_val, AbstractRange) || isa(param_val, Array)
            push!(sel_params, rand(param_val))
        else
            push!(sel_params, param_val)
        end
    end
    sel_params = tuple(sel_params...)
    (name=sel_name, params=sel_params, indicator=rand(1:num_indicators))
end

function init_strategy_data(num_indicators::Int, num_selectors::Int)
    indicators = Array{NamedTuple,1}()
    selectors = Array{NamedTuple,1}()
    for i in 1:num_indicators
        ind = get_rand_indicator()
        push!(indicators, ind)
    end
    for i in 1:num_selectors
        sel = get_rand_selector(num_indicators)
        push!(selectors, sel)
    end
    StrategyData(indicators, selectors, OrderedDict())
end

function init_strategy_data(num_indicators::Int, num_selectors::Int, symbols::Array{String,1})
    indicators = Array{NamedTuple,1}()
    selectors = Array{NamedTuple,1}()
    for i in 1:num_indicators
        ind = get_rand_indicator(symbols)
        push!(indicators, ind)
    end
    for i in 1:num_selectors
        sel = get_rand_selector(num_indicators)
        push!(selectors, sel)
    end
    StrategyData(indicators, selectors, OrderedDict())
end

function mutate_strategy_data!(strat_data::StrategyData, ind_mut_rate::Float64, sel_mut_rate::Float64)
    for i in eachindex(strat_data.indicators)
        if rand() < ind_mut_rate
            strat_data.indicators[i] = get_rand_indicator()
            continue
        end
        ind_name = strat_data.indicators[i].name
        ind_params = Number[]
        for j in eachindex(indicators_params[ind_name])
            if rand() < ind_mut_rate
                param_val = indicators_params[ind_name][j]
                if isa(param_val, AbstractRange) || isa(param_val, Array)
                    push!(ind_params, rand(param_val))
                else
                    push!(ind_params, param_val)
                end
            else
                push!(ind_params, strat_data.indicators[i].params[j])
            end
        end
        ind_params = tuple(ind_params...)
        strat_data.indicators[i] = (name=ind_name, params=ind_params)
    end
    num_indicators = length(strat_data.indicators)
    for i in eachindex(strat_data.selectors)
        if rand() < sel_mut_rate
            strat_data.selectors[i] = get_rand_selector(num_indicators)
            continue
        end
        sel_name = strat_data.selectors[i].name
        sel_params = Number[]
        for j in eachindex(selectors_params[sel_name])
            if rand() < sel_mut_rate
                param_val = selectors_params[sel_name][j]
                if isa(param_val, AbstractRange) || isa(param_val, Array)
                    push!(sel_params, rand(param_val))
                else
                    push!(sel_params, param_val)
                end
            else
                push!(sel_params, strat_data.selectors[i].params[j])
            end
        end
        sel_params = tuple(sel_params...)
        if rand() < sel_mut_rate
            indicator = rand(1:num_indicators)
        else
            indicator = strat_data.selectors[i].indicator
        end
        strat_data.selectors[i] = (name=sel_name, params=sel_params, indicator=indicator)
    end
end

function mutate_strategy_data!(strat_data::StrategyData, ind_mut_rate::Float64, sel_mut_rate::Float64, symbols::Array{String,1})
    for i in eachindex(strat_data.indicators)
        if rand() < ind_mut_rate
            strat_data.indicators[i] = get_rand_indicator(symbols)
            continue
        end
        ind_name = strat_data.indicators[i].name
        ind_params = Number[]
        for j in eachindex(indicators_params[ind_name])
            if rand() < ind_mut_rate
                param_val = indicators_params[ind_name][j]
                if isa(param_val, AbstractRange) || isa(param_val, Array)
                    push!(ind_params, rand(param_val))
                else
                    push!(ind_params, param_val)
                end
            else
                push!(ind_params, strat_data.indicators[i].params[j])
            end
        end
        ind_params = tuple(ind_params...)
        if rand() < ind_mut_rate
            symbol = rand(symbols)
        else
            symbol = strat_data.indicators[i].symbol
        end
        strat_data.indicators[i] = (name=ind_name, params=ind_params, symbol=symbol)
    end
    num_indicators = length(strat_data.indicators)
    for i in eachindex(strat_data.selectors)
        if rand() < sel_mut_rate
            strat_data.selectors[i] = get_rand_selector(num_indicators)
            continue
        end
        sel_name = strat_data.selectors[i].name
        sel_params = Number[]
        for j in eachindex(selectors_params[sel_name])
            if rand() < sel_mut_rate
                param_val = selectors_params[sel_name][j]
                if isa(param_val, AbstractRange) || isa(param_val, Array)
                    push!(sel_params, rand(param_val))
                else
                    push!(sel_params, param_val)
                end
            else
                push!(sel_params, strat_data.selectors[i].params[j])
            end
        end
        sel_params = tuple(sel_params...)
        if rand() < sel_mut_rate
            indicator = rand(1:num_indicators)
        else
            indicator = strat_data.selectors[i].indicator
        end
        strat_data.selectors[i] = (name=sel_name, params=sel_params, indicator=indicator)
    end
end

function load_conv_rates(data_dir::String, symbol::String, time_frame::String, acc_curr::String, date_from=nothing, date_to=nothing)
    base_curr = symbol[1:3]
    quote_curr = symbol[4:6]
    all_symbols = replace.(filter(isdir, readdir(data_dir, join=true)), ["$data_dir/" => "",])
    # acc_quote_rate
    if acc_curr * quote_curr in all_symbols
        acc_quote_rate = load_csv_file(data_dir, acc_curr * quote_curr, time_frame, date_from - Week(1), date_to).raw_data[:, [:datetime, :open]]
    elseif quote_curr * acc_curr in all_symbols
        acc_quote_rate = load_csv_file(data_dir, quote_curr * acc_curr, time_frame, date_from - Week(1), date_to).raw_data[:, [:datetime, :open]]
        acc_quote_rate.open = 1 ./ acc_quote_rate.open
    elseif quote_curr == acc_curr
        acc_quote_rate = 1.
    else
        throw(error("Conv-rate error, $(acc_curr * quote_curr) or $(quote_curr * acc_curr) not found!"))
    end
    # acc_base_rate
    if acc_curr * base_curr in all_symbols
        acc_base_rate = load_csv_file(data_dir, acc_curr * base_curr, time_frame, date_from - Week(1), date_to).raw_data[:, [:datetime, :open]]
    elseif base_curr * acc_curr in all_symbols
        acc_base_rate = load_csv_file(data_dir, base_curr * acc_curr, time_frame, date_from - Week(1), date_to).raw_data[:, [:datetime, :open]]
        acc_base_rate.open = 1 ./ acc_base_rate.open
    elseif base_curr == acc_curr
        acc_base_rate = 1.
    else
        throw(error("Conv-rate error, $(acc_curr * base_curr) or $(base_curr * acc_curr) not found!"))
    end
    # usd_base_rate
    if "USD" * base_curr in all_symbols
        usd_base_rate = load_csv_file(data_dir, "USD" * base_curr, time_frame, date_from - Week(1), date_to).raw_data[:, [:datetime, :open]]
    elseif base_curr * "USD" in all_symbols
        usd_base_rate = load_csv_file(data_dir, base_curr * "USD", time_frame, date_from - Week(1), date_to).raw_data[:, [:datetime, :open]]
        usd_base_rate.open = 1 ./ usd_base_rate.open
    elseif base_curr == "USD"
        usd_base_rate = 1.
    else
        throw(error("Conv-rate error, $("USD" * base_curr) or $(base_curr * "USD") not found!"))
    end
    # acc_usd_rate
    if acc_curr * "USD" in all_symbols
        acc_usd_rate = load_csv_file(data_dir, acc_curr * "USD", time_frame, date_from - Week(1), date_to).raw_data[:, [:datetime, :open]]
    elseif "USD" * acc_curr in all_symbols
        acc_usd_rate = load_csv_file(data_dir, "USD" * acc_curr, time_frame, date_from - Week(1), date_to).raw_data[:, [:datetime, :open]]
        acc_usd_rate.open = 1 ./ acc_usd_rate.open
    elseif acc_curr == "USD"
        acc_usd_rate = 1.
    else
        throw(error("Conv-rate error, $(acc_curr * "USD") or $("USD" * acc_curr) not found!"))
    end
    return acc_quote_rate, acc_base_rate, usd_base_rate, acc_usd_rate
end

function get_raw_data_from(time_frame::String, sim_date_from::DateTime)
    if time_frame == "Daily"
        raw_data_from = sim_date_from - Month(30)
    elseif time_frame == "Hour12"
        raw_data_from = sim_date_from - Month(15)
    elseif time_frame == "Hour4"
        raw_data_from = sim_date_from - Month(5)
    elseif time_frame == "Hour"
        raw_data_from = sim_date_from - Month(2)
    elseif time_frame == "Minute15"
        raw_data_from = sim_date_from - Month(1)
    elseif time_frame == "Minute5"
        raw_data_from = sim_date_from - Day(15)
    elseif time_frame == "Minute"
        raw_data_from = sim_date_from - Day(5)
    end
    raw_data_from
end

function get_fitness(fitness::Vector, disc_rate::Float64=1., cv_coef::Float64=0.1)
    if fitness != ones(length(fitness))
        disc_rates = reverse([disc_rate^i for i in  1:length(fitness)])
        _fitness = log.(fitness) .+ 1
        mean_fit = mean(_fitness)
        disc_mean_fit = mean(1 .+ ((_fitness .- 1) .* disc_rates))
        disc_std_fit = sqrt(mean(broadcast(x::Float64 -> (x - mean_fit)^2, _fitness) .* disc_rate))
        cv = disc_std_fit / disc_mean_fit
        return disc_mean_fit - (cv_coef * cv)
    end
    return 1. # the evolution algo treat "1" as the worst possible results
end

# function float_index(x::Float64, values::Array)
#     indices = [1:length(values);]
#     dists = abs.(indices .- ((x * (length(values)-1) + 1)))
#     values[indices[argmin(dists)]]
# end
