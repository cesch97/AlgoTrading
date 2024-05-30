
include("modules/trad_system.jl")
include("modules/simulator.jl")
include("trad_sys_analysis.jl")
using DataStructures
using YAML
using Plots


function trad_sys_evaluation(config_file)
    if isa(config_file, String)
        config = YAML.load_file(config_file)
        log_dir = config["log_dir"]
    else
        config = config_file
        log_dir = config["evals_dir"]
    end
    log_name = config["log_name"]
    mkpath("$log_dir/$log_name")
    YAML.write_file("$log_dir/$log_name/$log_name.yml", config)

    data_dir = config["data_dir"]
    strats_dir = config["strats_dir"]
    trad_sys_dir = config["trad_sys_dir"]
    symbols = config["symbols"]
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
    # logging
    num_results = config["num_results"]
    analysis_conf_file = config["analysis_conf_file"]
    # misc
    cuda = config["cuda"]

    # load daily data of all symbols
    raw_data_from = get_raw_data_from("Hour12", sim_date_from)
    daily_raw_data = Dict()
    for symbol in symbols
        data = load_csv_file(data_dir, symbol, "Hour12", raw_data_from, sim_date_to).raw_data
        daily_raw_data[symbol] = data
    end

    # load all trad_systems
    all_trad_sys = readdir(trad_sys_dir * "/trad_systems", join=true, sort=true)
    filter!(x -> isfile(x), all_trad_sys)
    filter!(x -> occursin(".jld2", x), all_trad_sys)
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

    results = OrderedDict()
    all_fitness = Float64[]
    strats_trad_data = Dict()
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
            fitness, week_rets, week_perf = backtest_ts!(trad_data, trad_sys, init_balance, sim_params, 
                                                         sim_date_from, sim_date_to; train=false, force_init_balance=force_init_balance,
                                                         _strategies_reg=strategies_reg, _strats_data_reg=strats_data_reg, disc_rate=fit_disc_rate, cv_coef=fit_cv_coef)
            results[trad_sys_name] = (fitness=fitness, week_rets=week_rets, week_perf=week_perf)
            push!(all_fitness, fitness)
        end
    end
    ord_fit = reverse(sortperm(all_fitness))
    best_res_keys = collect(keys(results))[ord_fit][1:num_results]

    # plotting results
    plot(xlabel="Datetime", ylabel="Balance", legend=:topleft)
    (!force_init_balance) && plot!(yaxis=:log)
    for trad_sys in best_res_keys
        plot!(results[trad_sys].week_perf.datetime, results[trad_sys].week_perf.balance, label=trad_sys)
    end    
    savefig("$log_dir/$log_name/performance.png")
    rets_sub_plots = Any[]
    for trad_sys in best_res_keys
        rets_plot = plot(results[trad_sys].week_rets.datetime, results[trad_sys].week_rets.ret, legend=false)
        hline!([0 for _ in 1:nrow(results[trad_sys].week_rets)])
        push!(rets_sub_plots, rets_plot)
    end
    plot(rets_sub_plots..., layout=(num_results, 1))
    savefig("$log_dir/$log_name/returns.png")

    # running trad-sys_analysis on the best results
    if isnothing(analysis_conf_file)
        project_paths = YAML.load_file("./configs/paths.yml")
        if ENV["USER"] == "ec2-user"
            project_paths = project_paths["ec2"]
        else
            project_paths = project_paths["local"]
        end
        fx_data = YAML.load_file("./configs/fx_data.yml")
        analysis_conf_file = YAML.load_file("./configs/trad_sys_analysis.yml")
        analysis_conf_file = merge(project_paths, fx_data, analysis_conf_file)
    end
    for trad_sys in best_res_keys
        analysis_conf_file["analysis_dir"] = "$log_dir/$log_name"
        analysis_conf_file["log_name"] = trad_sys
        analysis_conf_file["trad_sys_name"] = trad_sys
        trad_sys_analysis(analysis_conf_file)
    end
end
