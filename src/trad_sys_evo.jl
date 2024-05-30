
include("modules/evolution.jl")
using YAML


function evolve_trad_sys(config_file::String, strats_trad_data=Dict())
    config = YAML.load_file(config_file)
    data_dir = config["data_dir"]
    strats_dir = config["strats_dir"]
    symbols = config["symbols"]
    time_frames = config["time_frames"]
    # strategy
    excl_strats = config["excl_strats"]
    num_indicators = config["num_indicators"]
    num_selectors = config["num_selectors"]
    num_strategies = config["num_strategies"]
    force_diff_symbs = config["force_diff_symbs"]
    # neural network
    h_layers = config["h_layers"]
    weight_decay = config["weight_decay"]
    cuda = config["cuda"]
    # GA
    pop_size = config["pop_size"]
    num_epochs = config["num_epochs"]
    tourn_size = config["tourn_size"]
    elitism = config["elitism"]
    cross_rate = config["cross_rate"]
    ind_mut_rate = config["ind_mut_rate"]
    sel_mut_rate = config["sel_mut_rate"]
    strats_mut_rate = config["strats_mut_rate"]
    model_mut_σ = config["model_mut_σ"]
    lr = config["lr"]
    fit_disc_rate = config["fit_disc_rate"]
    fit_cv_coef = config["fit_cv_coef"]
    # simulation
    sim_date_from = config["sim_date_from"]
    sim_date_to = config["sim_date_to"]
    init_balance = config["init_balance"]
    acc_curr = config["acc_curr"]
    risk = config["risk"]
    spread = config["spread"]
    leverage = config["leverage"]
    commission = config["commission"]
    # logging
    log_name = config["log_name"]
    log_dir = config["log_dir"]
    log_every = config["log_every"]

    all_strats = readdir(strats_dir * "/strategies", join=true, sort=true)
    filter!(x -> isfile(x), all_strats)
    filter!(x -> occursin(".jld2", x), all_strats)
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
    _all_strats = String[]
    all_strats_symbs = String[]
    for strat_name in all_strats
        strategy = load_strategy(strats_dir * "/strategies/" * strat_name * ".jld2")
        if strategy != false
            symbol = strategy.symbol
            time_frame = strategy.time_frame
            if (symbol in symbols) && (time_frame in time_frames)
                push!(_all_strats, strat_name)
                push!(all_strats_symbs, symbol)
            end
        end
    end
    all_strats = _all_strats

    raw_data_from = get_raw_data_from("Hour12", sim_date_from)
    daily_raw_data = Dict()
    for symbol in symbols
        data = load_csv_file(data_dir, symbol, "Hour12", raw_data_from, sim_date_to).raw_data
        daily_raw_data[symbol] = data
    end

    trad_data = TradData(symbols, "Hour12", daily_raw_data, OrderedDict(), OrderedDict(), OrderedDict())

    best_trad_sys, best_fitness, strats_trad_data = genetic_algo!(data_dir, strats_dir,
                                                                  pop_size, num_epochs, tourn_size, cross_rate, ind_mut_rate, sel_mut_rate, strats_mut_rate,
                                                                  elitism, model_mut_σ, lr, fit_disc_rate, fit_cv_coef, num_indicators, num_selectors, num_strategies, trad_data,
                                                                  h_layers, weight_decay, cuda, symbols, all_strats, all_strats_symbs, force_diff_symbs, 
                                                                  init_balance, risk, spread, leverage, commission,
                                                                  log_name, log_dir, log_every, 
                                                                  sim_date_from, sim_date_to, strats_trad_data)
    best_trad_sys, best_fitness, strats_trad_data
end