
include("data_preprocessing.jl")
include("model.jl")
include("strategy.jl")
include("simulator.jl")
using Printf
import Base: copy
using JLD2


mutable struct TradSystem
    strat_data::StrategyData
    model
    strategies::Array{String,1}
end


function init_trad_system(symbols::Array{String,1}, num_strats::Int, num_indicators::Int, num_selectors::Int, model, 
                          all_strats::Array{String,1}, all_strats_symbs::Array{String,1}, force_diff_symbs::Bool)
    strat_data = init_strategy_data(num_indicators, num_selectors, symbols)
    strategies = String[]
    strat_symbs = String[]
    while length(strategies) < num_strats
        i = rand(1:length(all_strats))
        strat = all_strats[i]
        symb = all_strats_symbs[i]
        if (force_diff_symbs) && !(strat in strategies) && !(symb in strat_symbs)
            push!(strategies, strat)
            push!(strat_symbs, symb)
        elseif !force_diff_symbs && !(strat in strategies)
            push!(strategies, strat)
            push!(strat_symbs, symb)
        end
    end
    TradSystem(strat_data, model, strategies)
end

function load_strat_raw_data(strategy::Strategy, data_dir::String, sim_date_from=nothing, sim_date_to=nothing)
    symbol = strategy.symbol
    time_frame = strategy.time_frame
    raw_data_from = get_raw_data_from(time_frame, sim_date_from)
    strat_trad_data = load_csv_file(data_dir, symbol, time_frame, raw_data_from, sim_date_to)
    acc_quote_rate, acc_base_rate, usd_base_rate, acc_usd_rate = load_conv_rates(data_dir, symbol, time_frame, "EUR", sim_date_from, sim_date_to)
    conv_rate = (acc_quote_rate=acc_quote_rate, acc_base_rate=acc_base_rate, usd_base_rate=usd_base_rate, acc_usd_rate=acc_usd_rate)
    strat_trad_data, conv_rate
end

@everywhere function make_strat_trad_data!(strategy::Strategy, strat_raw_data::TradData, sim_date_from=nothing, sim_date_to=nothing, cuda::Bool=false)
    cuda && (strategy.model = to_cuda(strategy.model))
    update_trad_data!(strat_raw_data, strategy.strat_data; force_std=false)
    trad_df, ind_matrix = make_sim_data(strat_raw_data, sim_date_from, sim_date_to)
    signal = forward(strategy.model, ind_matrix; arg_max=true)
    trad_df.signal = signal
    sim_data_weeks = DataFrame[]
    last_dt = trad_df[1, :].datetime
    if dayofweek(last_dt) == 7
        last_week = week(last_dt) + 1
    else
        last_week = week(last_dt)
    end
    last_dt_id = 1
    for i in 2:nrow(trad_df)
        dt = trad_df[i, :].datetime
        if (week(dt) != last_week) && (day(dt) != day(last_dt))
            push!(sim_data_weeks, trad_df[last_dt_id: i-1, :])
            last_dt_id = i
            last_dt = trad_df[i, :].datetime
            last_week = week(last_dt)
        elseif (dayofweek(dt) == 7) && (day(dt) != day(last_dt))
            push!(sim_data_weeks, trad_df[last_dt_id: i-1, :])
            last_dt_id = i
            last_dt = trad_df[i, :].datetime
            last_week = week(last_dt) + 1
        end
    end
    if last_dt_id != nrow(trad_df)
        push!(sim_data_weeks, trad_df[last_dt_id: nrow(trad_df), :])
    end
    sim_data_weeks
end

# sim_ind_data = nothing
# sim_sel_data = nothing
# sim_std_data = nothing

function backtest_ts!(trad_data::TradData, trad_sys::TradSystem, init_balance::Number, sim_params::SimulatorParams,
                      sim_date_from=nothing, sim_date_to=nothing; train::Bool=true, force_init_balance::Bool=true,
                      _strategies_reg=nothing, _strats_data_reg=nothing, disc_rate::Float64=1., cv_coef::Float64=0.1,
                      analysis_dir=nothing, analysis_name=nothing)
    if train
        global strategies_reg, strats_data_reg
        update_trad_data!(trad_data, trad_sys.strat_data; force_std=true)
    else
        update_trad_data!(trad_data, trad_sys.strat_data; force_std=false)
    end
    datetime, mat = make_sim_data(trad_data, sim_date_from, sim_date_to)

    # global sim_ind_data, sim_sel_data, sim_std_data

    # df = DataFrame()
    # ind_data = DataFrame(trad_data.ind_data)
    # ind_data.datetime = trad_data.raw_data[collect(keys(trad_data.raw_data))[1]][!, :datetime]
    # last_dt = ind_data[1, :datetime]
    # if dayofweek(last_dt) == 7
    #     last_week = week(last_dt) + 1
    # else
    #     last_week = week(last_dt)
    # end
    # for i in 2:nrow(ind_data)
    #     dt = ind_data[i, :datetime]
    #     if (week(dt) != last_week) && (day(dt) != day(last_dt))
    #         push!(df, ind_data[i, :])
    #         last_dt = ind_data[i, :datetime]
    #         last_week = week(last_dt)
    #     elseif (dayofweek(dt) == 7) && (day(dt) != day(last_dt))
    #         push!(df, ind_data[i, :])
    #         last_dt = ind_data[i, :datetime]
    #         last_week = week(last_dt) + 1
    #     end
    # end       
    # df = slice_by_time(df, sim_date_from-Day(1), sim_date_to)      
    # sim_ind_data = copy(Matrix{Float32}(select(df, Not(:datetime))))

    # df = DataFrame()
    # sel_data = DataFrame(trad_data.sel_data)
    # sel_data.datetime = trad_data.raw_data[collect(keys(trad_data.raw_data))[1]][!, :datetime]
    # last_dt = sel_data[1, :datetime]
    # if dayofweek(last_dt) == 7
    #     last_week = week(last_dt) + 1
    # else
    #     last_week = week(last_dt)
    # end
    # for i in 2:nrow(sel_data)
    #     dt = sel_data[i, :datetime]
    #     if (week(dt) != last_week) && (day(dt) != day(last_dt))
    #         push!(df, sel_data[i, :])
    #         last_dt = sel_data[i, :datetime]
    #         last_week = week(last_dt)
    #     elseif (dayofweek(dt) == 7) && (day(dt) != day(last_dt))
    #         push!(df, sel_data[i, :])
    #         last_dt = sel_data[i, :datetime]
    #         last_week = week(last_dt) + 1
    #     end
    # end       
    # df = slice_by_time(df, sim_date_from-Day(1), sim_date_to)      
    # sim_sel_data = copy(Matrix{Float32}(select(df, Not(:datetime))))

    # sim_std_data = mat

    allocation = forward(trad_sys.model, mat; arg_max=false)
    
    fits = Float64[]
    balance = init_balance
    week_rets = DataFrame()
    week_perf = DataFrame()
    # analysis
    trad_logs = NamedTuple[]
    signal_logs = NamedTuple[]
    conv_rates_logs = NamedTuple[]
    allc_logs = NamedTuple[]
    acc_logs = NamedTuple[]

    for (i, dt) in enumerate(datetime)
        dt < sim_date_from && continue
        ret = 0
        for j in eachindex(trad_sys.strategies)
            strat_name = trad_sys.strategies[j]
            strategy = train ? strategies_reg[strat_name] : _strategies_reg[strat_name]
            symbol = strategy.symbol
            strat_trad_data = train ? strats_data_reg[strat_name] : _strats_data_reg[strat_name]
            sim_data_weeks = strat_trad_data.sim_data_weeks
            sim_params.acc_quote_ratio = strat_trad_data.acc_quote_rate
            sim_params.acc_base_ratio = strat_trad_data.acc_base_rate
            sim_params.usd_base_ratio = strat_trad_data.usd_base_rate
            sim_params.acc_usd_ratio = strat_trad_data.acc_usd_rate
            sim_params.init_balance = allocation[i, j + 1] * (((train) || (force_init_balance)) ? init_balance : balance)
            if occursin("JPY", symbol)
                pip_size = 0.01
            else
                pip_size = 0.0001
            end
            sim_params.pip_size = pip_size
            last_bar = (open=0., high=0., low=0., close=0.)
            sim_status = SimulatorStatus(DateTime(0), 0., 0., sim_params.init_balance, sim_params.init_balance, Position[], 0., 1., last_bar)
            simulation = Simulation(strategy.symbol, strategy.time_frame, sim_params, strategy, sim_data_weeks[i], sim_status, [], [], 0)
            if (sim_params.init_balance > 0) || (!isnothing(analysis_name))
                if !isnothing(analysis_name)
                    logs = backtest!(simulation; force_init_balance=true, analysis=true)
                else
                    backtest!(simulation; force_init_balance=true, analysis=false)
                end
            end
            final_balance = simulation.sim_status.balance
            strat_ret = final_balance - sim_params.init_balance
            ret += strat_ret
            # analysis
            if !isnothing(analysis_name)
                trad_logs = [trad_logs; merge.([(strategy=j,),], simulation.trade_logs)]
                signal_logs = [signal_logs; merge.([(strategy=j,),], logs.signal_logs)]
                conv_rates_logs = [conv_rates_logs; merge.([(strategy=j,),], logs.conv_rates_logs)]
            end
        end
        balance = max(balance, 1.)
        if (train) || (force_init_balance)
            push!(fits, max(1 + (ret / init_balance), 1e-6))
            push!(week_rets, (datetime = dt, ret=(ret/init_balance)))
        else
            push!(fits, max(1 + (ret / balance), 1e-6))
            push!(week_rets, (datetime = dt, ret=(ret/balance)))
        end
        balance += ret
        push!(week_perf, (datetime = dt, balance = balance))
        # analysis
        (!isnothing(analysis_name)) && (allc_logs = [allc_logs; tuple(dt, allocation[i, :]...)])
    end

    # analysis
    if !isnothing(analysis_name)   
        trad_logs = DataFrame(trad_logs)
        sort!(trad_logs, [:close_time,])
        signal_logs = DataFrame(signal_logs)
        sort!(signal_logs, [:datetime,])
        conv_rates_logs = DataFrame(conv_rates_logs)
        sort!(conv_rates_logs, [:datetime,])
        allc_logs = DataFrame(allc_logs)

        balances = fill(NaN, length(trad_sys.strategies) + 1)
        equities = fill(NaN, length(trad_sys.strategies) + 1)
        margins = fill(NaN, length(trad_sys.strategies) + 1)
        allc_week = nothing
        balance = init_balance
        for i in 1:nrow(trad_logs)
            dt = trad_logs[i, :open_time]
            if (week(dt) != allc_week)
                allc = filter(x -> x[1] <= dt, allc_logs)[end, :]
                allc_dt = allc[1]
                allc = convert(Array{Float64,1}, allc[2:end])
                if dayofweek(allc_dt) == 7
                    allc_week = week(allc_dt) + 1
                else
                    allc_week = week(allc_dt)
                end
                balances = allc .* balance
                equities = allc .* balance
                margins = fill(0., length(trad_sys.strategies) + 1)
            end
            strategy = trad_logs[i, :strategy]
            balances[strategy + 1] += trad_logs[i, :profit]
            equities[strategy + 1] = balances[strategy + 1] + (trad_logs[i, :equity] - trad_logs[i, :balance])
            margins[strategy + 1] = trad_logs[i, :margin_used]
            balance=sum(balances)
            push!(acc_logs, (datetime=trad_logs[i, :close_time], balance=balance, equities=sum(equities), margin_used=sum(margins)))
        end
        acc_logs = DataFrame(acc_logs)

        analysis_path = analysis_dir * "/" * analysis_name
        mkpath(analysis_path)
        CSV.write(analysis_path * "/trad_logs.csv", trad_logs)
        CSV.write(analysis_path * "/signal_logs.csv", signal_logs)
        CSV.write(analysis_path * "/conv_rates_logs.csv", conv_rates_logs)
        CSV.write(analysis_path * "/allc_logs.csv", allc_logs)
        CSV.write(analysis_path * "/acc_logs.csv", acc_logs)
    end

    fitness = get_fitness(fits, disc_rate, cv_coef) 
    fitness, week_rets, week_perf
end

function copy(trad_sys::TradSystem)
    strat_data = StrategyData(deepcopy(trad_sys.strat_data.indicators),
                              deepcopy(trad_sys.strat_data.selectors),
                              deepcopy(trad_sys.strat_data.std_params))
    model = copy(trad_sys.model)
    strategies = copy(trad_sys.strategies)
    TradSystem(strat_data, model, strategies)
end

function dump_trad_sys(trad_sys::TradSystem, file::String)
    strat_data = trad_sys.strat_data
    if isa(trad_sys.model, Array{LinearLayer,1})
        cpu_model = trad_sys.model
    else
        cpu_model = to_cpu(trad_sys.model)
    end
    strategies = trad_sys.strategies
    @save file strat_data cpu_model strategies
end

function load_trad_sys(file::String)
    @load file strat_data cpu_model strategies
    TradSystem(strat_data, cpu_model, strategies)
end

function mutate_trad_sys!(trad_sys::TradSystem,  ind_mut_rate::Float64, sel_mut_rate::Float64, strats_mut_rate::Float64, 
                          symbols::Array{String,1}, all_strats::Array{String,1}, all_strats_symbs::Array{String,1}, force_diff_symbs::Bool)
    mutate_strategy_data!(trad_sys.strat_data, ind_mut_rate, sel_mut_rate, symbols)

    strats_symbs = String[]
    if force_diff_symbs
        for i in eachindex(trad_sys.strategies)
            strat = trad_sys.strategies[i]
            strat_idx = findall(x -> x==strat, all_strats)[1]
            strat_symb = all_strats_symbs[strat_idx]
            push!(strats_symbs, strat_symb)
        end
    end

    for i in eachindex(trad_sys.strategies)
        if rand() < strats_mut_rate
            while true
                j = rand(1:length(all_strats))
                strat = all_strats[j]
                symb = all_strats_symbs[j]
                if (force_diff_symbs) && !(strat in trad_sys.strategies) && !(symb in strats_symbs)
                    trad_sys.strategies[i] = strat
                    strats_symbs[i] = symb
                    break
                elseif !force_diff_symbs && !(strat in trad_sys.strategies)
                    trad_sys.strategies[i] = strat
                    break
                end
            end
        end
    end
end

function trad_sys_crossover!(parent_1::TradSystem, parent_2::TradSystem, cross_rate::Float64,
                             all_strats::Array{String,1}, all_strats_symbs::Array{String,1}, force_diff_symbs::Bool)
    for i in eachindex(parent_1.strat_data.selectors)
        if rand() < cross_rate
            parent_1.strat_data.selectors[i] = parent_2.strat_data.selectors[i]
            ind_id = parent_1.strat_data.selectors[i].indicator
            parent_1.strat_data.indicators[ind_id] = parent_2.strat_data.indicators[ind_id]
        end
    end

    strats_symbs = String[]
    if force_diff_symbs
        for i in eachindex(parent_1.strategies)
            strat = parent_1.strategies[i]
            strat_idx = findall(x -> x==strat, all_strats)[1]
            strat_symb = all_strats_symbs[strat_idx]
            push!(strats_symbs, strat_symb)
        end
    end

    for i in eachindex(parent_1.strategies)
        if rand() < cross_rate
            strat = parent_2.strategies[i]
            strat_idx = findall(x -> x==strat, all_strats)[1]
            strat_symb = all_strats_symbs[strat_idx]

            if (force_diff_symbs) && !(strat in parent_1.strategies) && !(strat_symb in strats_symbs)
                parent_1.strategies[i] = strat
                strats_symbs[i] = strat_symb
            elseif !force_diff_symbs && !(strat in parent_1.strategies)
                parent_1.strategies[i] = strat
            end
        end
    end
end
