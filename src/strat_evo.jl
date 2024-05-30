
include("modules/data_preprocessing.jl")
include("modules/model.jl")
include("modules/strategy.jl")
include("modules/simulator.jl")
include("modules/evolution.jl")
using YAML

function evolve_strategy(config_file::String)
    config = YAML.load_file(config_file)
    data_dir = config["data_dir"]
    symbol = config["symbol"]
    time_frame = config["time_frame"]
    # strategy
    num_indicators = config["num_indicators"]
    num_selectors = config["num_selectors"]
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
    params_mut_rate = config["params_mut_rate"]
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
    
    # Data Loading
    raw_data_from = get_raw_data_from(time_frame, sim_date_from)
    trad_data = load_csv_file(data_dir, symbol, time_frame, raw_data_from, sim_date_to)

    # Simulation
    acc_quote_rate, acc_base_rate, usd_base_rate, acc_usd_rate = load_conv_rates(data_dir, symbol, time_frame, acc_curr, sim_date_from, sim_date_to)

    if occursin("JPY", symbol)
        pip_size = 0.01
    else
        pip_size = 0.0001
    end
    sim_params = SimulatorParams(sim_date_from, sim_date_to, init_balance, risk, spread, leverage, 
                                acc_quote_rate, acc_base_rate, usd_base_rate, acc_usd_rate, commission, pip_size)

    best_strategy, best_fitness = genetic_algo!(pop_size, num_epochs, tourn_size, cross_rate, ind_mut_rate, sel_mut_rate, params_mut_rate, 
                                                elitism, model_mut_σ, lr, fit_disc_rate, fit_cv_coef, 
                                                num_indicators, num_selectors, trad_data, sim_params, 
                                                h_layers, weight_decay, cuda,
                                                log_name, log_dir, log_every)
    best_strategy, best_fitness
end