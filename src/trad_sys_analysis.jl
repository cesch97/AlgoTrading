
using YAML
using Distributed
include("modules/trad_system.jl")
include("modules/simulator.jl")
using DataStructures
using YAML
using Plots
using StatsPlots

function trad_sys_analysis(config_file)
    if isa(config_file, String)
        config = YAML.load_file(config_file)
    else
        config = config_file
    end
    
    data_dir = config["data_dir"]
    strats_dir = config["strats_dir"]
    trad_sys_dir = config["trad_sys_dir"]
    analysis_dir = config["analysis_dir"]
    symbols = config["symbols"]
    trad_sys_name = config["trad_sys_name"]
    log_name = config["log_name"]
    # simulation
    sim_date_from = config["sim_date_from"]
    sim_date_to = config["sim_date_to"]
    init_balance = config["init_balance"]
    acc_curr = config["acc_curr"]
    risk = config["risk"]
    spread = config["spread"]
    leverage = config["leverage"]
    commission = config["commission"]
    force_init_balance = config["force_init_balance"]
    fit_disc_rate = config["fit_disc_rate"]
    fit_cv_coef = config["fit_cv_coef"]
    # misc
    cuda = config["cuda"]

    # load daily data of all symbols
    raw_data_from = get_raw_data_from("Hour12", sim_date_from)
    daily_raw_data = Dict()
    for symbol in symbols
        data = load_csv_file(data_dir, symbol, "Hour12", raw_data_from, sim_date_to).raw_data
        daily_raw_data[symbol] = data
    end

    strategies_reg = Dict()
    strats_data_reg = Dict()
    sim_params = SimulatorParams(sim_date_from, sim_date_to, init_balance, risk, spread, leverage, 1., 1., 1., 1., commission, 0.0001)

    trad_data = TradData(symbols, "Hour12", deepcopy(daily_raw_data), OrderedDict(), OrderedDict(), OrderedDict())
    trad_sys = load_trad_sys("$trad_sys_dir/trad_systems/$trad_sys_name.jld2")
    (cuda) && (trad_sys.model = to_cuda(trad_sys.model))
    valid = true
    for strat_name in trad_sys.strategies
        if !haskey(strategies_reg, strat_name)
            strategy = load_strategy(strats_dir * "/strategies/" * strat_name * ".jld2")
            if strategy != false
                strategies_reg[strat_name] = strategy
                strat_raw_data, conv_rates = load_strat_raw_data(strategy, data_dir, sim_date_from, sim_date_to)
                strat_trad_data = make_strat_trad_data!(strategy, strat_raw_data, sim_date_from, sim_date_to, cuda)
                strats_data_reg[strat_name] = (sim_data_weeks=strat_trad_data, conv_rates...)
            else
                valid = false
            end
        end
    end
    if valid
        backtest_ts!(trad_data, trad_sys, init_balance, sim_params, 
                     sim_date_from, sim_date_to; train=false, force_init_balance=force_init_balance,
                     _strategies_reg=strategies_reg, _strats_data_reg=strats_data_reg, disc_rate=fit_disc_rate, cv_coef=fit_cv_coef,
                     analysis_dir=analysis_dir, analysis_name=log_name)
    else
        throw(error("Invalid trad_system!"))
    end
    analysis_path = analysis_dir * "/" * log_name
    
    trad_logs = CSV.File(analysis_path * "/trad_logs.csv") |> DataFrame
    signal_logs = CSV.File(analysis_path * "/signal_logs.csv") |> DataFrame
    conv_rates_logs = CSV.File(analysis_path * "/conv_rates_logs.csv") |> DataFrame
    allc_logs = CSV.File(analysis_path * "/allc_logs.csv") |> DataFrame
    acc_logs = CSV.File(analysis_path * "/acc_logs.csv") |> DataFrame

    plot(xlabel="Datetime", ylabel="Balance", legend=:topleft)
    (!force_init_balance) && plot!(yaxis=:log)
    plot!(acc_logs.datetime, acc_logs.balance)  
    savefig("$analysis_path/balance.png")
    
    allc_bars = Array{Float64,2}(undef, nrow(allc_logs), ncol(allc_logs)-1)
    for i in 2:ncol(allc_logs)
        for j in 1:nrow(allc_logs)
            allc_bars[j, i-1] = allc_logs[j, i]
        end
    end
    groupedbar(allc_bars, bar_position=:stack, legend=false, ticks=false)
    savefig("$analysis_path/allocation.png")

    trad_sys_dict = OrderedDict("strat_data" => OrderedDict("indicators" => OrderedDict(), "selectors" => OrderedDict(), "std_params" => trad_sys.strat_data.std_params),
                                "strategies" => trad_sys.strategies)
    for (i, indicator) in enumerate(trad_sys.strat_data.indicators)
        ind_dict = OrderedDict("name" => indicator.name, "params" => [indicator.params...], "symbol" => indicator.symbol)
        trad_sys_dict["strat_data"]["indicators"][string(i)] = ind_dict
    end
    for (i, selector) in enumerate(trad_sys.strat_data.selectors)
        sel_dict = OrderedDict("name" => selector.name, "params" => [selector.params...], "indicator" => selector.indicator)
        trad_sys_dict["strat_data"]["selectors"][string(i)] = sel_dict
    end
    YAML.write_file("$analysis_path/trad_sys.yml", trad_sys_dict)

    rm("$analysis_path/strategies"; force=true, recursive=true)
    mkpath("$analysis_path/strategies")
    for strat_name in trad_sys.strategies
        strategy = strategies_reg[strat_name]
        strat_dict = OrderedDict("symbol" => strategy.symbol, "time_frame" => strategy.time_frame,
                                 "strat_data" => OrderedDict("indicators" => OrderedDict(), "selectors" => OrderedDict(), "std_params" => strategy.strat_data.std_params),
                                 "params" => strategy.params)
        for (i, indicator) in enumerate(strategy.strat_data.indicators)
            ind_dict = OrderedDict("name" => indicator.name, "params" => [indicator.params...])
            strat_dict["strat_data"]["indicators"][string(i)] = ind_dict
        end
        for (i, selector) in enumerate(strategy.strat_data.selectors)
            sel_dict = OrderedDict("name" => selector.name, "params" => [selector.params...], "indicator" => selector.indicator)
            strat_dict["strat_data"]["selectors"][string(i)] = sel_dict
        end
        YAML.write_file("$analysis_path/strategies/$strat_name.yml", strat_dict)
    end    

    trad_logs, signal_logs, conv_rates_logs, allc_logs, acc_logs
end

# project_paths = YAML.load_file("./configs/paths.yml")
# if ENV["USER"] == "ec2-user"
#     project_paths = project_paths["ec2"]
# else
#     project_paths = project_paths["local"]
# end
# fx_data = YAML.load_file("./configs/fx_data.yml")

# conf_file = YAML.load_file("./configs/trad_sys_analysis.yml")
# conf_file = merge(project_paths, fx_data, conf_file)
# trad_logs, signal_logs, conv_rates_logs, allc_logs, acc_logs = trad_sys_analysis(conf_file)

# # # # # # # # # # #

# trad_sys = load_trad_sys("/data/AlgoTrading/trad_systems/trad_systems/trad-sys_creat_A-46.jld2")
# strat = load_strategy("/data/AlgoTrading/strategies/strategies/strat_creat_A-74.jld2")