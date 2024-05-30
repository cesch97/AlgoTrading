

@everywhere include("src/modules/data_preprocessing.jl")
@everywhere include("src/modules/model.jl")
@everywhere include("src/modules/strategy.jl")
@everywhere include("src/modules/trad_system.jl")
@everywhere include("src/modules/simulator.jl")
using Printf
import Base:copy
using JLD2

# Structures stored localy on processes in the "worker_reg" dictionary

@everywhere mutable struct StratWorker
    trad_data::TradData
    sim_params::SimulatorParams
    strategy::Strategy
end

@everywhere mutable struct TsWorker
    trad_data::TradData
    trad_sys::TradSystem
    sim_params::NamedTuple
end

#

# registries stored localy on processes

@everywhere worker_reg = Dict() # key: individual id "pop[i]", value: worker

@everywhere strategies_reg = Dict() # key: strat_name, value: ::Strategy
@everywhere strats_data_reg = Dict() # key: strat_name, value: ::Tuple(DataFrame) (weekly dfs for strat backtesting and conv. rates)

#

function init_proc_work_reg(pop_size::Int)
    # create a dictionary with the process id as key
    # and an array of int (individual ids) as value
    proc_work_reg = Dict()
    proc_id = 2
    for i in 1:pop_size
        if !haskey(proc_work_reg, proc_id)
            proc_work_reg[proc_id] = [i,]
        else
            push!(proc_work_reg[proc_id], i)
        end
        proc_id += 1
        (proc_id > nprocs()) && (proc_id = 2) 
    end
    proc_work_reg
end

@everywhere function set_worker_reg(_worker_reg::Dict)
    global worker_reg # it's alredy in the process
    delete!.([worker_reg,], keys(worker_reg)) # cleaning up
    for key in keys(_worker_reg)
        worker_reg[key] = _worker_reg[key]
    end
end

function load_work_to_procs(pop::Union{Array{StratWorker}, Array{TsWorker}}, proc_work_reg::Dict)
    # make a dictionary with the individual id as key 
    # and the worker as value and set as global variable
    # in the target process
    @sync begin
        for proc_id in 2:nprocs()
            @async begin
                worker_reg = Dict()
                for i in proc_work_reg[proc_id]
                    worker_reg[i] = pop[i]
                end
                remotecall_wait(set_worker_reg, proc_id, worker_reg)
            end
        end
    end
end

@everywhere function rss_objective!(sim_params::SimulatorParams, strategy::Strategy, trad_data::TradData; train::Bool=true, disc_rate::Float64=1., cv_coef::Float64=0.1)
    if train
        update_trad_data!(trad_data, strategy.strat_data; force_std=true)
    else
        update_trad_data!(trad_data, strategy.strat_data; force_std=false)
    end

    trad_df, ind_matrix = make_sim_data(trad_data, sim_params.date_from, sim_params.date_to)
    signal = forward(strategy.model, ind_matrix; arg_max=true)
    trad_df.signal = signal

    sim_data_weeks = DataFrame[]
    last_dt = trad_df[1, :].datetime
    last_dt_id = 1
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
        push!(sim_data_weeks, trad_df[last_dt_id:nrow(trad_df), :])
    end

    fitness = Array{Float64,1}(undef, length(sim_data_weeks))
    for i in eachindex(sim_data_weeks)
        last_bar = (open = 0., high = 0., low = 0., close = 0.)
        sim_status = SimulatorStatus(DateTime(0), 0., 0., sim_params.init_balance, sim_params.init_balance, Position[], 0., 1., last_bar)
        simulation = Simulation(strategy.symbol, strategy.time_frame, sim_params, strategy, sim_data_weeks[i], sim_status, [], [], 0)
        backtest!(simulation; force_init_balance=true)
        final_balance = simulation.sim_status.balance
        init_balance = simulation.sim_params.init_balance
        max_drawdown = simulation.max_drawdown
        fit = final_balance / init_balance
        fitness[i] = max(fit, 1e-6)
    end
    for i in eachindex(fitness)
        if fitness[i] == 1.
            fitness[i] = minimum(fitness)
        end
    end
    get_fitness(fitness, disc_rate, cv_coef)
end

function copy(worker::StratWorker)
    trad_data = TradData(worker.trad_data.symbol,
                         worker.trad_data.time_frame,
                         worker.trad_data.raw_data,
                         deepcopy(worker.trad_data.ind_data),
                         deepcopy(worker.trad_data.sel_data),
                         deepcopy(worker.trad_data.std_data))
    strat_data = StrategyData(deepcopy(worker.strategy.strat_data.indicators),
                              deepcopy(worker.strategy.strat_data.selectors),
                              deepcopy(worker.strategy.strat_data.std_params))
    params = deepcopy(worker.strategy.params)
    strategy = Strategy(worker.strategy.symbol, worker.strategy.time_frame, strat_data, worker.strategy.model, params)
    StratWorker(trad_data, worker.sim_params, strategy)
end

function copy(worker::TsWorker)
    trad_data = TradData(worker.trad_data.symbol,
                         worker.trad_data.time_frame,
                         worker.trad_data.raw_data,
                         deepcopy(worker.trad_data.ind_data),
                         deepcopy(worker.trad_data.sel_data),
                         deepcopy(worker.trad_data.std_data))
    strat_data = StrategyData(deepcopy(worker.trad_sys.strat_data.indicators),
                              deepcopy(worker.trad_sys.strat_data.selectors),
                              deepcopy(worker.trad_sys.strat_data.std_params))
    strategies = copy(worker.trad_sys.strategies)
    trad_sys = TradSystem(strat_data, worker.trad_sys.model, strategies)
    TsWorker(trad_data, trad_sys, worker.sim_params)
end

function init_pop(pop_size::Int, model_pop, sim_params::SimulatorParams, trad_data::TradData, 
                  num_inidcators::Int, num_selectors::Int)
    pop = StratWorker[]
    for i in 1:pop_size
        w_trad_data = TradData(trad_data.symbol,
                               trad_data.time_frame,
                               trad_data.raw_data,
                               deepcopy(trad_data.ind_data),
                               deepcopy(trad_data.sel_data),
                               deepcopy(trad_data.std_data))
        strategy = init_strategy(trad_data.symbol, trad_data.time_frame, num_inidcators, num_selectors, model_pop[i])
        worker = StratWorker(w_trad_data, sim_params, strategy)
        push!(pop, worker)
    end
    pop
end

function init_pop(symbols::Array{String,1}, all_strats::Array{String,1}, all_strats_symbs::Array{String,1}, force_diff_symbs::Bool,
                  pop_size::Int, model_pop, trad_data::TradData, 
                  num_indicators::Int, num_selectors::Int, num_strategies::Int, sim_params::NamedTuple)
    pop = TsWorker[]
    for i in 1:pop_size
        w_trad_data = TradData(trad_data.symbol,
                        trad_data.time_frame,
                        trad_data.raw_data,
                        deepcopy(trad_data.ind_data),
                        deepcopy(trad_data.sel_data),
                        deepcopy(trad_data.std_data))
        trad_sys = init_trad_system(symbols, num_strategies, num_indicators, num_selectors, model_pop[i], all_strats, all_strats_symbs, force_diff_symbs)
        worker = TsWorker(w_trad_data, trad_sys, sim_params)
        push!(pop, worker)
    end
    pop
end

# Functions run on the processes

@everywhere function run_strat_process(workers_id::Array{Int}, strategies::Array{Strategy}, disc_rate::Float64=1., cv_coef::Float64=0.1)
    global worker_reg
    fitnesses = Float64[]
    std_params = OrderedDict[]
    for i in eachindex(workers_id)
        worker = worker_reg[workers_id[i]]
        worker.strategy = strategies[i]
        fitness = rss_objective!(worker.sim_params, worker.strategy, worker.trad_data; train=true, disc_rate=disc_rate, cv_coef=cv_coef)
        push!(fitnesses, fitness)
        push!(std_params, worker.strategy.strat_data.std_params)
    end
    fitnesses, std_params
end

@everywhere function run_ts_process(workers_id::Array{Int}, trad_systems::Array{TradSystem}, strats_to_add::Dict, strats_data_to_add::Dict, ts_sim_ps, disc_rate::Float64=1., cv_coef::Float64=0.1)
    global worker_reg, strategies_reg, strats_data_reg

    # updating registries on process
    for strat_name in keys(strats_to_add)
        strategies_reg[strat_name] = strats_to_add[strat_name]
        strats_data_reg[strat_name] = strats_data_to_add[strat_name]
    end
    
    fitnesses = Float64[]
    std_params = OrderedDict[]
    for i in eachindex(workers_id)
        worker = worker_reg[workers_id[i]]
        worker.trad_sys = trad_systems[i]

        data_dir = ts_sim_ps.data_dir
        init_balance = ts_sim_ps.init_balance
        risk = ts_sim_ps.risk
        spread = ts_sim_ps.spread
        leverage = ts_sim_ps.leverage
        commission = ts_sim_ps.commission
        sim_date_from = ts_sim_ps.sim_date_from
        sim_date_to = ts_sim_ps.sim_date_to
        sim_params = SimulatorParams(sim_date_from, sim_date_to, init_balance, risk, spread, leverage, 1., 1., 1., 1., commission, 0.0001)
        fitness = backtest_ts!(worker.trad_data, worker.trad_sys, init_balance, sim_params,
                            sim_date_from, sim_date_to; train=true, force_init_balance=true, disc_rate=disc_rate, cv_coef=cv_coef)[1]
        push!(fitnesses, fitness)
        push!(std_params, worker.trad_sys.strat_data.std_params)
    end
    fitnesses, std_params
end

#

# Function that handles the mapping on the processes

function pmap!(pop::Array{StratWorker}, proc_work_reg::Dict, disc_rate::Float64=1., cv_coef::Float64=0.1)
    # spwan the strategy backtestings across the processes and retrieve the "fitness" and the new "std_params"
    fitness = Array{Float64,1}(undef, length(pop))
    futures = Future[]
    for proc_id in 2:nprocs()
        workers_id = Int[]
        strategies = Strategy[]
        for i in proc_work_reg[proc_id]
            push!(workers_id, i)
            push!(strategies, pop[i].strategy)
        end
        future = remotecall(run_strat_process, proc_id, workers_id, strategies, disc_rate, cv_coef)
        push!(futures, future)
    end 
    @sync begin
        for proc_id in 2:nprocs()
            @async begin
                fitnesses, std_params = fetch(futures[proc_id - 1])
                finalize(futures[proc_id - 1]) # memory leak -> https://discourse.julialang.org/t/understanding-distributed-memory-garbage-collection/8726
                for (i, j) in enumerate(proc_work_reg[proc_id])
                    fitness[j] = fitnesses[i]
                    pop[j].strategy.strat_data.std_params = std_params[i]
                end
            end
        end
    end
    fitness
end

function update_local_regs!(pop::Array{TsWorker}, strategies::Dict, strats_trad_data::Dict, data_dir::String, strats_dir::String, sim_date_from=nothing, sim_date_to=nothing, cuda=false)
    # update the local registries with the new strategies rquired for the epoch
    strats_raw_data = TradData[]
    strats_name = String[]
    strats = Strategy[]
    conv_rates = NamedTuple[]
    valid = true
    for i in eachindex(pop)
        trad_sys = pop[i].trad_sys
        for strat_name in trad_sys.strategies
            if !haskey(strategies, strat_name)
                strategy = load_strategy(strats_dir * "/strategies/" * strat_name * ".jld2")
                if strategy != false
                    strategies[strat_name] = strategy
                    strat_raw_data, conv_rate = load_strat_raw_data(strategy, data_dir, sim_date_from, sim_date_to)
                    if !haskey(strats_trad_data, strats_name)
                        push!(strats_raw_data, strat_raw_data)
                        push!(strats_name, strat_name)
                        push!(strats, strategy)
                        push!(conv_rates, conv_rate)
                    end
                else
                    valid = false
                end
            end                
        end
    end
    if !valid
        return false
    end

    if length(strats_raw_data) > 0
        if length(strats_raw_data) > 1
            _strats_trad_data = pmap(make_strat_trad_data!, strats, strats_raw_data, [sim_date_from for _ in eachindex(strats_raw_data)], [sim_date_to for _ in eachindex(strats_raw_data)], [cuda for _ in eachindex(strats_raw_data)])
            for i in eachindex(_strats_trad_data)
                strats_trad_data[strats_name[i]] = (sim_data_weeks=_strats_trad_data[i], conv_rates[i]...)
            end
        else
            # if there's only one do the task locally
            _strats_trad_data = make_strat_trad_data!(strats[1], strats_raw_data[1], sim_date_from, sim_date_to, cuda)
            strats_trad_data[strats_name[1]] = (sim_data_weeks=_strats_trad_data, conv_rates[1]...)
        end
    end

    return true
end

function pmap!(pop::Array{TsWorker}, proc_work_reg::Dict, procs_strat_reg::Dict, strategies::Dict, strats_trad_data::Dict, ts_sim_ps::NamedTuple, disc_rate::Float64=1., cv_coef::Float64=0.1)
    fitness = Array{Float64}(undef, length(pop))
    futures = Future[]
    for proc_id in 2:nprocs()
        workers_id = Int[]
        trad_systems = TradSystem[]
        strats_to_send = Dict()
        strats_data_to_send = Dict()
        for i in proc_work_reg[proc_id]
            push!(workers_id, i)
            trad_sys = pop[i].trad_sys
            push!(trad_systems, trad_sys)
            for strat_name in trad_sys.strategies
                if !(strat_name in procs_strat_reg[proc_id])
                    if !haskey(strats_to_send, strat_name)
                        strats_to_send[strat_name] = strategies[strat_name]
                        strats_data_to_send[strat_name] = strats_trad_data[strat_name]
                    end
                    push!(procs_strat_reg[proc_id], strat_name)
                end
            end
        end
        future = remotecall(run_ts_process, proc_id, workers_id, trad_systems, strats_to_send, strats_data_to_send, ts_sim_ps, disc_rate, cv_coef)
        push!(futures, future)
    end
    @sync begin
        for proc_id in 2:nprocs()
            @async begin
                fitnesses, std_params = fetch(futures[proc_id - 1])
                finalize(futures[proc_id - 1]) # memory leak -> https://discourse.julialang.org/t/understanding-distributed-memory-garbage-collection/8726
                for (i, j) in enumerate(proc_work_reg[proc_id])
                    fitness[j] = fitnesses[i]
                    pop[j].trad_sys.strat_data.std_params = std_params[i]
                end
            end
        end
    end
    fitness
end


#

function save_checkpoint(num_epoch::Int, log_name::String, log_dir::String, 
                         pop::Array{StratWorker,1}, model::AbstractArray, experts_reg::Array{Strategy,1}, max_fit_reg::Array{Float64})
    checkpoint_file_name = "$log_dir/$log_name-checkpoint.jld2"
    if isa(model, Array{CuLinearLayer,1})
        model_cpu = to_cpu(model)
    else
        model_cpu = model
    end
    @save checkpoint_file_name num_epoch pop model_cpu experts_reg max_fit_reg
end

function save_checkpoint(num_epoch::Int, log_name::String, log_dir::String, 
    pop::Array{TsWorker,1}, model::AbstractArray, 
    experts_reg::Array{TradSystem,1}, max_fit_reg::Array{Float64})
    checkpoint_file_name = "$log_dir/$log_name-checkpoint.jld2"
    if isa(model, Array{CuLinearLayer,1})
        model_cpu = to_cpu(model)
    else
        model_cpu = model
    end
    @save checkpoint_file_name num_epoch pop model_cpu experts_reg max_fit_reg
end

function load_strat_checkpoint(log_name::String, log_dir::String)
    checkpoint_file_name = "$log_dir/$log_name-checkpoint.jld2"
    if "$log_name-checkpoint.jld2" in readdir(log_dir, join=false)
        @load checkpoint_file_name num_epoch pop model_cpu experts_reg max_fit_reg
        return num_epoch, pop, model_cpu, experts_reg, max_fit_reg
    end
    false
end

function load_ts_checkpoint(log_name::String, log_dir::String)
    checkpoint_file_name = "$log_dir/$log_name-checkpoint.jld2"
    if "$log_name-checkpoint.jld2" in readdir(log_dir, join=false)
        @load checkpoint_file_name num_epoch pop model_cpu experts_reg max_fit_reg
        return num_epoch, pop, model_cpu, experts_reg, max_fit_reg
    end
    false
end

function genetic_algo!(pop_size::Int, epochs::Int, tourn_size::Int, cross_rate::Float64, ind_mut_rate::Float64, sel_mut_rate::Float64, params_mut_rate::Float64,
                       elitism::Int, σ_model::Float64, lr::Float64, fit_disc_rate::Float64, fit_cv_coef::Float64,
                       num_indicators::Int, num_selectors::Int, trad_data::TradData, sim_params::SimulatorParams,
                       h_layers::Array{Int,1}, weight_decay::Float64, cuda::Bool,
                       log_name::String, log_dir::String, log_every::Int)
                       
    checkpoint = load_strat_checkpoint(log_name, log_dir)
    if checkpoint == false
        model = init_model(num_selectors, 5, h_layers, cuda)
        model_pop = [copy(model) for i in 1:pop_size]
        model_exp_noise = get_exp_noise.(model_pop, [σ_model,])
        model_pop = mutate.(model_pop, model_exp_noise)
        pop = init_pop(pop_size, model_pop, sim_params, trad_data, num_indicators, num_selectors)
        experts_reg = Strategy[copy(pop[1].strategy)]
        max_fit_reg = Float64[-1.]
        num_epoch = 0
        init_num_epoch = 1
    else
        num_epoch, pop, model, experts_reg, max_fit_reg = checkpoint
        if cuda
            model = to_cuda(model)
        end
        model_pop = [copy(model) for i in 1:pop_size]
        model_exp_noise = get_exp_noise.(model_pop, [σ_model,])
        model_pop = mutate.(model_pop, model_exp_noise)
        init_num_epoch = num_epoch + 1
        for i in 1:pop_size
            pop[i].strategy.model = model_pop[i]
            pop[i].sim_params = sim_params
        end
    end
    txt_log_file = "$log_dir/logs/$log_name.txt"
    file_name = log_dir * "/" * log_name * ".jld2"
    max_fit = 0
    mean_fit = 0
    proc_work_reg = init_proc_work_reg(pop_size)
    load_work_to_procs(pop, proc_work_reg)
    for epoch in init_num_epoch:epochs
        num_epoch = epoch
        fitness = pmap!(pop, proc_work_reg, fit_disc_rate, fit_cv_coef)
        for i in eachindex(fitness) # Hack to disincentive doing nothing
            if fitness[i] == 1
                fitness[i] = minimum(fitness)
            end
        end
        max_fit = maximum(fitness)
        mean_fit = mean(fitness)
        if (max_fit > max_fit_reg[end]) && (max_fit != 1.)
            push!(max_fit_reg, max_fit)
            push!(experts_reg, copy(pop[argmax(fitness)].strategy))
        end
        if epoch % log_every == 0
            @printf("%i -> best-result: %.3f | gen-best: %.3f, gen-mean: %.3f\n", epoch, max_fit_reg[end], max_fit, mean_fit)
            open(txt_log_file, "a") do io
                write(io, "$epoch -> best-result: $(max_fit_reg[end]) | gen-best: $(max_fit), gen-mean: $(mean_fit)\n")
            end
            if max_fit_reg[end] != -1.
                dump_strategy(experts_reg[end], file_name)
            end
            save_checkpoint(num_epoch, log_name, log_dir, pop, model, experts_reg, max_fit_reg)
            GC.gc() # found memory leak, this could help
        end
        advantage = Float32(-1) .* Float32.((fitness .- mean_fit) ./ (max(std(fitness), 1e-6)))
        grad = compute_grad_approx(model_exp_noise, advantage)
        sgd_update!(model, grad, lr, σ_model, pop_size, weight_decay)
        ord_fit = reverse(sortperm(fitness))
        fitness = fitness[ord_fit]
        pop = pop[ord_fit]
        new_pop = pop[1:elitism]
        for i in elitism + 1:pop_size
            sel_pool = [i,]
            fit_pool = [fitness[i],]
            pop_pool = filter(x -> x != i, [1:pop_size;])
            for j in 1:tourn_size - 1
                sel = rand(pop_pool)
                filter!(x -> x != sel, pop_pool)
                push!(sel_pool, sel)
                push!(fit_pool, fitness[sel])
            end
            parent_1 = copy(pop[sel_pool[argmax(fit_pool)]])
            pop_pool = filter(x -> x != sel_pool[argmax(fit_pool)], [1:pop_size;])
            sel_pool = Int[]
            fit_pool = Float64[]
            for j in 1:tourn_size
                sel = rand(pop_pool)
                filter!(x -> x != sel, pop_pool)
                push!(sel_pool, sel)
                push!(fit_pool, fitness[sel])
            end
            parent_2 = pop[sel_pool[argmax(fit_pool)]]
            strategy_crossover!(parent_1.strategy, parent_2.strategy, cross_rate)
            mutate_strategy!(parent_1.strategy, ind_mut_rate, sel_mut_rate, params_mut_rate)
            push!(new_pop, parent_1)
        end
        pop = new_pop
        model_pop = [copy(model) for i in 1:pop_size]
        model_exp_noise = get_exp_noise.(model_pop, [σ_model,])
        model_pop = mutate.(model_pop, model_exp_noise)
        for i in 1:pop_size
            pop[i].strategy.model = model_pop[i]
        end
    end
    @printf("%i -> best-result: %.3f | gen-best: %.3f, gen-mean: %.3f\n", num_epoch, max_fit_reg[end], max_fit, mean_fit)
    open(txt_log_file, "a") do io
        write(io, "$num_epoch -> best-result: $(max_fit_reg[end]) | gen-best: $(max_fit), gen-mean: $(mean_fit)\n")
    end
    if max_fit_reg[end] != -1.
        dump_strategy(experts_reg[end], file_name)
    end
    rm("$log_dir/$log_name-checkpoint.jld2", force=true)
    experts_reg[end], max_fit_reg[end]    
end

function genetic_algo!(data_dir::String, strats_dir::String,
                       pop_size::Int, epochs::Int, tourn_size::Int, cross_rate::Float64, ind_mut_rate::Float64, sel_mut_rate::Float64, strats_mut_rate::Float64,
                       elitism::Int, σ_model::Float64, lr::Float64, fit_disc_rate::Float64, fit_cv_coef::Float64,
                       num_indicators::Int, num_selectors::Int, num_strategies::Int, trad_data::TradData,
                       h_layers::Array{Int,1}, weight_decay::Float64, cuda::Bool,
                       symbols::Array{String,1}, all_strats::Array{String,1}, all_strats_symbs::Array{String,1}, force_diff_symbs::Bool,
                       init_balance::Number, risk::Float64, spread::Dict, leverage::Dict, commission::Float64,
                       log_name::String, log_dir::String, log_every::Int,
                       sim_date_from=nothing, sim_date_to=nothing, strats_trad_data=Dict())

    ts_sim_ps = (data_dir = data_dir, init_balance = init_balance, risk = risk, spread = spread, leverage = leverage, commission = commission,
                 sim_date_from = sim_date_from, sim_date_to = sim_date_to)
    checkpoint = load_ts_checkpoint(log_name, log_dir)
    if checkpoint == false
        # model = init_model(num_selectors, num_strategies + 1, h_layers, cuda)
        model = init_model(num_selectors, num_strategies, h_layers, cuda)
        model_pop = [copy(model) for i in 1:pop_size]
        model_exp_noise = get_exp_noise.(model_pop, [σ_model,])
        model_pop = mutate.(model_pop, model_exp_noise)
        pop = init_pop(symbols, all_strats,  all_strats_symbs, force_diff_symbs, pop_size, model_pop, trad_data, num_indicators, num_selectors, num_strategies, ts_sim_ps)
        experts_reg = TradSystem[copy(pop[1].trad_sys)]
        max_fit_reg = Float64[-1.]
        num_epoch = 0
        init_num_epoch = 1
    else
        num_epoch, pop, model, experts_reg, max_fit_reg  = checkpoint
        if cuda
            model = to_cuda(model)
        end
        model_pop = [copy(model) for i in 1:pop_size]
        model_exp_noise = get_exp_noise.(model_pop, [σ_model,])
        model_pop = mutate.(model_pop, model_exp_noise)
        init_num_epoch = num_epoch + 1
        for i in 1:pop_size
            pop[i].trad_sys.model = model_pop[i]
            pop[i].sim_params = ts_sim_ps
        end
    end
    txt_log_file = "$log_dir/logs/$log_name.txt"
    file_name = log_dir * "/" * log_name * ".jld2"

    max_fit = 0
    mean_fit = 0

    proc_work_reg = init_proc_work_reg(pop_size)
    load_work_to_procs(pop, proc_work_reg)
    strategies = Dict()
    # strats_trad_data = Dict()
    procs_strat_reg = Dict()
    for i in 2:nprocs()
        procs_strat_reg[i] = String[]
    end

    for epoch in init_num_epoch:epochs
        num_epoch = epoch
        ret = update_local_regs!(pop, strategies, strats_trad_data, data_dir, strats_dir, sim_date_from, sim_date_to, cuda)
        (!ret) && (break) # if it fails to load a strategy then interrupt the evolution (eg. a strat used is been cleaned up) 
        
        fitness = pmap!(pop, proc_work_reg, procs_strat_reg, strategies, strats_trad_data, ts_sim_ps, fit_disc_rate, fit_cv_coef)
        for i in eachindex(fitness)
            if fitness[i] == 1
                fitness[i] = minimum(fitness)
            end
        end

        max_fit = maximum(fitness)
        mean_fit = mean(fitness)
        if (max_fit > max_fit_reg[end]) && (max_fit != 1.)
            push!(max_fit_reg, max_fit)
            push!(experts_reg, copy(pop[argmax(fitness)].trad_sys))
        end
        if epoch % log_every == 0
            @printf("%i -> best-result: %.3f | gen-best: %.3f, gen-mean: %.3f\n", epoch, max_fit_reg[end], max_fit, mean_fit)
            open(txt_log_file, "a") do io
                write(io, "$epoch -> best-result: $(max_fit_reg[end]) | gen-best: $(max_fit), gen-mean: $(mean_fit)\n")
            end
            if max_fit_reg[end] != -1.
                dump_trad_sys(experts_reg[end], file_name)
            end
            save_checkpoint(num_epoch, log_name, log_dir, pop, model, experts_reg, max_fit_reg)
            GC.gc() # found memory leak, this could help
        end
        advantage = Float32(-1) .* Float32.((fitness .- mean_fit) ./ (max(std(fitness), 1e-6)))
        grad = compute_grad_approx(model_exp_noise, advantage)
        sgd_update!(model, grad, lr, σ_model, pop_size, weight_decay)
        ord_fit = reverse(sortperm(fitness))
        fitness = fitness[ord_fit]
        pop = pop[ord_fit]
        new_pop = pop[1:elitism]
        for i in elitism + 1:pop_size
            sel_pool = [i,]
            fit_pool = [fitness[i],]
            pop_pool = filter(x -> x != i, [1:pop_size;])
            for j in 1:tourn_size - 1
                sel = rand(pop_pool)
                filter!(x -> x != sel, pop_pool)
                push!(sel_pool, sel)
                push!(fit_pool, fitness[sel])
            end
            parent_1 = copy(pop[sel_pool[argmax(fit_pool)]])
            pop_pool = filter(x -> x != sel_pool[argmax(fit_pool)], [1:pop_size;])
            sel_pool = Int[]
            fit_pool = Float64[]
            for j in 1:tourn_size
                sel = rand(pop_pool)
                filter!(x -> x != sel, pop_pool)
                push!(sel_pool, sel)
                push!(fit_pool, fitness[sel])
            end
            parent_2 = pop[sel_pool[argmax(fit_pool)]]
            trad_sys_crossover!(parent_1.trad_sys, parent_2.trad_sys, cross_rate, all_strats, all_strats_symbs, force_diff_symbs)
            mutate_trad_sys!(parent_1.trad_sys, ind_mut_rate, sel_mut_rate, strats_mut_rate, symbols, all_strats, all_strats_symbs, force_diff_symbs) 
            push!(new_pop, parent_1)
        end
        pop = new_pop
        model_pop = [copy(model) for i in 1:pop_size]
        model_exp_noise = get_exp_noise.(model_pop, [σ_model,])
        model_pop = mutate.(model_pop, model_exp_noise)
        for i in 1:pop_size
            pop[i].trad_sys.model = model_pop[i]
        end
    end
    @printf("%i -> best-result: %.3f | gen-best: %.3f, gen-mean: %.3f\n", num_epoch, max_fit_reg[end], max_fit, mean_fit)
    open(txt_log_file, "a") do io
        write(io, "$num_epoch -> best-result: $(max_fit_reg[end]) | gen-best: $(max_fit), gen-mean: $(mean_fit)\n")
    end
    if max_fit_reg[end] != -1.
        dump_trad_sys(experts_reg[end], file_name)
    end
    rm("$log_dir/$log_name-checkpoint.jld2", force=true)
    experts_reg[end], max_fit_reg[end], strats_trad_data    
end
