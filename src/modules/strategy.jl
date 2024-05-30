include("data_preprocessing.jl")
include("model.jl")
using JLD2
import Base: copy


mutable struct Strategy
    symbol::String
    time_frame::String
    strat_data::StrategyData
    model::Union{Array{LinearLayer}, Array{CuLinearLayer}}
    params::Dict
end

strat_params = Dict(
    "sl" => 5:100,
    "tp_1" => 5:100,
    "tp_2" => 5:100,
    "time_limit" => 15:1440
)

function get_strat_params(param_name::String)
    if isa(strat_params[param_name], AbstractRange) || isa(strat_params[param_name], Array)
        param = rand(strat_params[param_name])
    else
        param = strat_params[param_name]
    end
    param
end

function init_strategy(symbol::String, time_frame::String, num_indicators::Int, num_selectors::Int, model)
    strat_data = init_strategy_data(num_indicators, num_selectors)
    params=Dict()
    for param_name in keys(strat_params)
        param_params = get_strat_params(param_name)
        params[param_name] = param_params
    end
    Strategy(symbol, time_frame, strat_data, model, params)
end

function mutate_strat_params!(strategy::Strategy, mut_rate::Float64)
    for param_name in keys(strategy.params)
        if rand() < mut_rate
            strategy.params[param_name] = get_strat_params(param_name)
        end
    end
end

function mutate_strategy!(strategy::Strategy,  ind_mut_rate::Float64, sel_mut_rate::Float64, params_mut_rate::Float64)
    mutate_strategy_data!(strategy.strat_data, ind_mut_rate, sel_mut_rate)
    mutate_strat_params!(strategy, params_mut_rate)
end

function strategy_crossover!(parent_1::Strategy, parent_2::Strategy, cross_rate::Float64)
    for i in eachindex(parent_1.strat_data.selectors)
        if rand() < cross_rate
            parent_1.strat_data.selectors[i] = parent_2.strat_data.selectors[i]
            ind_id = parent_1.strat_data.selectors[i].indicator
            parent_1.strat_data.indicators[ind_id] = parent_2.strat_data.indicators[ind_id]
        end
    end
    for param_name in keys(parent_1.params)
        if rand() < cross_rate
            parent_1.params[param_name] = parent_2.params[param_name]
        end
    end
end

function dump_strategy(strategy::Strategy, file::String)
    symbol = strategy.symbol
    time_frame = strategy.time_frame
    strat_data = strategy.strat_data
    if isa(strategy.model, Array{LinearLayer,1})
        cpu_model = strategy.model
    else
        cpu_model = to_cpu(strategy.model)
    end
    params = strategy.params
    @save file symbol time_frame strat_data cpu_model params
end

function load_strategy(file::String)
    if isfile(file)
        @load file symbol time_frame strat_data cpu_model params
        return Strategy(symbol, time_frame, strat_data, cpu_model, params)
    else
        return false # if no strategy is found (e.g. was cleaned up)
    end
end

function copy(strategy::Strategy)
    strat_data = StrategyData(deepcopy(strategy.strat_data.indicators),
                              deepcopy(strategy.strat_data.selectors),
                              deepcopy(strategy.strat_data.std_params))
    if isa(strategy.model, Array{LinearLayer,1})
        model = copy(strategy.model)
    else
        model = to_cpu(strategy.model)
    end
    params = deepcopy(strategy.params)
    Strategy(strategy.symbol, strategy.time_frame, strat_data, model, params)
end
