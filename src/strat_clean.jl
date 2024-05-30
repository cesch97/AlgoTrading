
include("modules/simulator.jl")
include("modules/evolution.jl")
using YAML

function clean_strats(config_file)
    if isa(config_file, String)
        config = YAML.load_file(config_file)
        log_dir = config["log_dir"]
    else
        config = config_file
        log_dir = config["clean_dir"] * "/strategies"
    end
    log_name = config["log_name"]
    YAML.write_file("$log_dir/$log_name.yml", config)

    data_dir = config["data_dir"]
    strats_dir = config["strats_dir"]
    symbols = config["symbols"]
    time_frames = config["time_frames"]
    # strategy
    excl_strats = config["excl_strats"]
    # simulation
    sim_date_from = config["sim_date_from"]
    sim_date_to = config["sim_date_to"]
    init_balance = config["init_balance"]
    acc_curr = config["acc_curr"]
    risk = config["risk"]
    spread = config["spread"]
    leverage = config["leverage"]
    commission = config["commission"]
    fit_disc_rate = config["fit_disc_rate"]
    fit_cv_coef = config["fit_cv_coef"]
    # cleaning
    num_results = config["num_results"]
    # misc
    cuda = config["cuda"]

    all_strats = readdir(strats_dir * "/strategies", join=true, sort=true)
    filter!(x -> isfile(x), all_strats)
    filter!(x -> occursin(".jld2", x), all_strats)
    # Cleaning config and log files of strategies that did nothing in training
    all_configs = replace.(readdir(strats_dir * "/configs", join=false, sort=true), [".yml" => "",])
    all_logs = replace.(readdir(strats_dir * "/strategies/logs", join=false, sort=true), [".txt" => "",])
    _all_strats = replace.(all_strats, [strats_dir * "/strategies/" => "",])
    _all_strats = replace.(_all_strats, [".jld2" => "",])
    _all_strats = replace.(_all_strats, ["-checkpoint" => "",])
    for _config_file in all_configs
        if !(_config_file in _all_strats)
            rm(strats_dir * "/configs/" * _config_file * ".yml")
        end
    end
    for _log_file in all_logs
        if !(_log_file in _all_strats)
            rm(strats_dir * "/strategies/logs/" * _log_file * ".txt")
        end
    end
    #   
    check_points = filter(x -> occursin("-checkpoint", x), all_strats)
    check_points = replace.(check_points, ["-checkpoint" => "",])
    filter!(x -> !(x in check_points), all_strats)
    filter!(x -> !occursin("-checkpoint", x), all_strats)
    all_strats = replace.(all_strats, [strats_dir * "/strategies/" => "",])
    all_strats = replace.(all_strats, [".jld2" => "",])
    if !isnothing(excl_strats)
        for strat in excl_strats
            filter!(x -> !occursin(strat, x), all_strats)
        end
    end    

    if length(all_strats) > num_results
        strat_names = String[]
        strat_fits = Float64[]
        for strat_name in all_strats
            strategy = load_strategy(strats_dir * "/strategies/" * strat_name * ".jld2")
            symbol = strategy.symbol
            time_frame = strategy.time_frame
            if (symbol in symbols) && (time_frame in time_frames)
                cuda && (strategy.model = to_cuda(strategy.model))
                raw_data_from = get_raw_data_from(time_frame, sim_date_from)
                trad_data = load_csv_file(data_dir, symbol, time_frame, raw_data_from, sim_date_to)
                acc_quote_rate, acc_base_rate, usd_base_rate, acc_usd_rate = load_conv_rates(data_dir, symbol, time_frame, acc_curr, sim_date_from, sim_date_to)
                if occursin("JPY", symbol)
                    pip_size = 0.01
                else
                    pip_size = 0.0001
                end
                sim_params = SimulatorParams(sim_date_from, sim_date_to, init_balance, risk, spread, leverage, 
                                            acc_quote_rate, acc_base_rate, usd_base_rate, acc_usd_rate, commission, pip_size)
                fitness = rss_objective!(sim_params, strategy, trad_data; train=false, disc_rate=fit_disc_rate, cv_coef=fit_cv_coef)
                push!(strat_names, strat_name)
                push!(strat_fits, fitness)
            end
        end
        ord_fit = reverse(sortperm(strat_fits))
        strat_names = strat_names[ord_fit]
        if length(strat_names) > num_results
            strats_to_clean = strat_names[num_results+1: end]
            for strat_name in strats_to_clean
                strat_file_name = strats_dir * "/strategies/" * strat_name * ".jld2"
                config_file_name = strats_dir * "/configs/" * strat_name * ".yml"
                log_file_name = strats_dir * "/strategies/logs/" * strat_name * ".txt"
                mv(strat_file_name, log_dir * "/strategies/" * strat_name * ".jld2", force=true)
                mv(config_file_name, log_dir * "/configs/" * strat_name * ".yml", force=true)
                mv(log_file_name, log_dir * "/strategies/logs/" * strat_name * ".txt", force=true)
            end
        end
    end
end