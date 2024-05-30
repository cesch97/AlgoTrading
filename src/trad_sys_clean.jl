
include("modules/trad_system.jl")
include("modules/simulator.jl")
using DataStructures
using YAML

function clean_trad_sys(config_file)
    if isa(config_file, String)
        config = YAML.load_file(config_file)
        log_dir = config["log_dir"]
    else
        config = config_file
        log_dir = config["clean_dir"] * "/trad_systems"
    end
    log_name = config["log_name"]
    YAML.write_file("$log_dir/$log_name.yml", config)
    
    data_dir = config["data_dir"]
    strats_dir = config["strats_dir"]
    trad_sys_dir = config["trad_sys_dir"]
    symbols = config["symbols"]
    # strategy
    excl_ts = config["excl_ts"]
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
    # cleaning
    num_results = config["num_results"]
    # misc
    cuda = config["cuda"]

    # load daily data of all symbols
    raw_data_from = get_raw_data_from("Hour12", sim_date_from)
    daily_raw_data = Dict()
    for symbol in symbols
        data = load_csv_file(data_dir, symbol, "Hour12", raw_data_from, sim_date_to).raw_data
        daily_raw_data[symbol] = data
    end

    all_trad_sys = readdir(trad_sys_dir * "/trad_systems", join=true, sort=true)
    filter!(x -> isfile(x), all_trad_sys)
    filter!(x -> occursin(".jld2", x), all_trad_sys)
    # Cleaning config and log files of strategies that did nothing in training
    all_configs = replace.(readdir(trad_sys_dir * "/configs", join=false, sort=true), [".yml" => "",])
    all_logs = replace.(readdir(trad_sys_dir * "/trad_systems/logs", join=false, sort=true), [".txt" => "",])
    _all_trad_sys = replace.(all_trad_sys, [trad_sys_dir * "/trad_systems/" => "",])
    _all_trad_sys = replace.(_all_trad_sys, [".jld2" => "",])
    _all_trad_sys = replace.(_all_trad_sys, ["-checkpoint" => "",])
    for _config_file in all_configs
        if !(_config_file in _all_trad_sys)
            rm(trad_sys_dir * "/configs/" * _config_file * ".yml")
        end
    end
    for _log_file in all_logs
        if !(_log_file in _all_trad_sys)
            rm(trad_sys_dir * "/trad_systems/logs/" * _log_file * ".txt")
        end
    end
    #   
    check_points = filter(x -> occursin("-checkpoint", x), all_trad_sys)
    check_points = replace.(check_points, ["-checkpoint" => "",])
    filter!(x -> !(x in check_points), all_trad_sys)
    filter!(x -> !occursin("-checkpoint", x), all_trad_sys)
    all_trad_sys = replace.(all_trad_sys, [trad_sys_dir * "/trad_systems/" => "",])
    all_trad_sys = replace.(all_trad_sys, [".jld2" => "",])
    if !isnothing(excl_ts)
        for ts in excl_ts
            filter!(x -> !occursin(ts, x), all_trad_sys)
        end
    end

    if length(all_trad_sys) > num_results
        trad_sys_names = String[]
        trad_sys_fits = Float64[]
        strategies_reg = Dict()
        strats_data_reg = Dict()
        sim_params = SimulatorParams(sim_date_from, sim_date_to, init_balance, risk, spread, leverage, 1., 1., 1., 1., commission, 0.0001)
        for trad_sys_name in all_trad_sys
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
                fitness = backtest_ts!(trad_data, trad_sys, init_balance, sim_params, 
                                      sim_date_from, sim_date_to; train=false, force_init_balance=force_init_balance,
                                      _strategies_reg=strategies_reg, _strats_data_reg=strats_data_reg, disc_rate=fit_disc_rate, cv_coef=fit_cv_coef)[1]
            else
                fitness = NaN
            end
            push!(trad_sys_names, trad_sys_name)
            push!(trad_sys_fits, fitness)
        end

        # remove trad_systems that use strategies that do not exist
        trad_sys_to_clean = String[]
        not_nan_ts = Int[]
        for i in eachindex(trad_sys_fits)
            if isnan(trad_sys_fits[i])
                push!(trad_sys_to_clean, trad_sys_names[i])
            else
                push!(not_nan_ts, i)
            end
        end
        trad_sys_names = trad_sys_names[not_nan_ts]
        trad_sys_fits = trad_sys_fits[not_nan_ts]

        ord_fit = reverse(sortperm(trad_sys_fits))
        trad_sys_names = trad_sys_names[ord_fit]
        if length(trad_sys_names) > num_results
            trad_sys_to_clean = [trad_sys_to_clean; trad_sys_names[num_results+1: end]]
        end
        for trad_sys_name in trad_sys_to_clean
            trad_sys_file_name = trad_sys_dir * "/trad_systems/" * trad_sys_name * ".jld2"
            config_file_name = trad_sys_dir * "/configs/" * trad_sys_name * ".yml"
            log_file_name = trad_sys_dir * "/trad_systems/logs/" * trad_sys_name * ".txt"
            mv(trad_sys_file_name, log_dir * "/trad_systems/" * trad_sys_name * ".jld2", force=true)
            mv(config_file_name, log_dir * "/configs/" * trad_sys_name * ".yml", force=true)
            mv(log_file_name, log_dir * "/trad_systems/logs/" * trad_sys_name * ".txt", force=true)
        end
    end
end