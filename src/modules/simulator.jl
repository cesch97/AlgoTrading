include("data_preprocessing.jl")
include("strategy.jl")


mutable struct SimulatorParams
    date_from::DateTime
    date_to::DateTime
    init_balance::Number
    risk::Float64
    spread::Dict
    leverage::Dict
    acc_quote_ratio
    acc_base_ratio
    usd_base_ratio
    acc_usd_ratio
    commission::Float64
    pip_size::Float64
end

mutable struct Position
    volume::Number
    dir::String
    entry_price::Float64
    sl_price::Float64
    tp_price::Float64
    entry_time::DateTime
    limit_time::DateTime
    margin::Number
end

mutable struct SimulatorStatus
    datetime::DateTime
    bid::Float64
    ask::Float64
    balance::Number
    equity::Number
    positions::Array{Position,1}
    margin_used::Float64
    margin_level::Float64
    last_bar::NamedTuple
end

mutable struct Simulation
    symbol::String
    time_frame::String
    sim_params::SimulatorParams
    strategy::Strategy
    data::DataFrame
    sim_status::SimulatorStatus
    acc_logs::Array{NamedTuple,1}
    trade_logs::Array{NamedTuple,1}
    max_drawdown::Float64
end

function init_simulation!(sim_params::SimulatorParams, strategy::Strategy, trad_data::TradData; train::Bool=true)
    if train
        update_trad_data!(trad_data, strategy.strat_data; force_std=true)
    else
        update_trad_data!(trad_data, strategy.strat_data; force_std=false)
    end
    trad_df, ind_matrix = make_sim_data(trad_data, sim_params.date_from, sim_params.date_to)
    signal = forward(strategy.model, ind_matrix; arg_max=true)
    trad_df.signal = signal
    last_bar = (open=0., high=0., low=0., close=0.)
    sim_status = SimulatorStatus(DateTime(0), 0., 0., sim_params.init_balance, sim_params.init_balance, Position[], 0., 1., last_bar)
    Simulation(trad_data.symbol, trad_data.time_frame, sim_params, strategy, trad_df, sim_status, [], [], 0)
end

function close_position!(pos::Position, pos_id::Int, sim_status::SimulatorStatus, 
                         acc_quote_ratio::Float64, usd_base_ratio::Float64, acc_usd_ratio::Float64,
                         commission::Float64, trade_logs::Array{NamedTuple,1}, close_price=nothing)
    profit = calc_profit(pos, sim_status, acc_quote_ratio, usd_base_ratio, acc_usd_ratio, commission, close_price)
    sim_status.balance += profit
    if isnothing(close_price)
        log_close_price = (pos.dir=="buy" ? sim_status.bid : sim_status.ask)
    else
        log_close_price = close_price
    end
    log = (open_time=pos.entry_time, close_time=sim_status.datetime, dir=pos.dir,
           entry_price=pos.entry_price, close_price=log_close_price,
           volume=pos.volume, profit=profit, margin=pos.margin,
           balance=sim_status.balance, equity=sim_status.equity, margin_used=sim_status.margin_used)
    push!(trade_logs, log)
    up_pos = Position[]
    for i in eachindex(sim_status.positions) # removing the position
        if i != pos_id
            push!(up_pos, sim_status.positions[i])
        end
    end
    sim_status.positions = up_pos
end

function close_all_pos!(sim_status::SimulatorStatus, 
                        acc_quote_ratio::Float64, usd_base_ratio::Float64, acc_usd_ratio::Float64,
                        commission::Float64, trade_logs::Array{NamedTuple,1})
    while length(sim_status.positions) > 0
        pos = sim_status.positions[1]
        close_position!(pos, 1, sim_status, acc_quote_ratio, usd_base_ratio, acc_usd_ratio, commission, trade_logs)
    end
    sim_status.equity = sim_status.balance
    sim_status.margin_used = 0.
    sim_status.margin_level = 1.
end

function calc_profit(pos::Position, sim_status::SimulatorStatus, 
                     acc_quote_ratio::Float64, usd_base_ratio::Float64, acc_usd_ratio::Float64,
                     commission::Float64, close_price=nothing)
    if isnothing(close_price)
        if pos.dir == "buy"
            profit = (sim_status.bid - pos.entry_price) * pos.volume
        else
            profit = (pos.entry_price - sim_status.ask) * pos.volume
        end
    else
        if pos.dir == "buy"
            profit = (close_price - pos.entry_price) * pos.volume
        else
            profit = (pos.entry_price - close_price) * pos.volume
        end
    end
    profit /= acc_quote_ratio
    profit -= (((pos.volume / usd_base_ratio) * commission) / acc_usd_ratio) * 2
    profit 
end

function in_last_bar(sim_status::SimulatorStatus, price::Number, spread::Number=0)
    last_bar = sim_status.last_bar
    if (last_bar.low + spread <= price) && (last_bar.high + spread >= price)
        return true
    end
    return false
end

function update_positions!(sim_status::SimulatorStatus, 
                           acc_quote_ratio::Float64, usd_base_ratio::Float64, acc_usd_ratio::Float64,
                           spread::Number, commission::Float64, trade_logs::Array{NamedTuple,1})
    sim_status.equity = sim_status.balance
    if (dayofweek(sim_status.datetime) == 5) && (hour(sim_status.datetime) >= 17) # the Hour4 last bar is on friday at 17:00/18:00 (UTC+0)
        close_all_pos!(sim_status, acc_quote_ratio, usd_base_ratio, acc_usd_ratio, commission, trade_logs)
        sim_status.equity = sim_status.balance
    elseif length(sim_status.positions) > 0
        tot_profit = 0
        tot_margin = 0
        j = 1
        while true   
            pos = sim_status.positions[j]
            to_close = false
            close_price = nothing
            log = false
            if sim_status.datetime >= pos.limit_time
                to_close = true
            elseif pos.dir == "buy"
                if in_last_bar(sim_status, pos.sl_price, 0.)
                    to_close = true
                    close_price = pos.sl_price
                elseif in_last_bar(sim_status, pos.tp_price, 0.)
                    to_close = true
                    close_price = pos.tp_price
                elseif sim_status.bid <= pos.sl_price
                    to_close = true
                elseif  sim_status.bid >= pos.tp_price
                    to_close = true
                end
            else 
                if in_last_bar(sim_status, pos.sl_price, spread)
                    to_close = true
                    close_price = pos.sl_price
                elseif in_last_bar(sim_status, pos.tp_price, spread)
                    to_close = true
                    close_price = pos.tp_price
                elseif sim_status.ask >= pos.sl_price
                    to_close = true
                elseif sim_status.ask <= pos.tp_price
                    to_close = true
                end
            end
            if to_close
                close_position!(pos, j, sim_status, acc_quote_ratio, usd_base_ratio, acc_usd_ratio, commission, trade_logs, close_price)
            else
                tot_margin += pos.margin
                profit = calc_profit(pos, sim_status, acc_quote_ratio, usd_base_ratio, acc_usd_ratio, commission)
                tot_profit += profit
                j += 1
            end
            if j > length(sim_status.positions)
                break
            end
        end
        sim_status.equity = sim_status.balance + tot_profit
        sim_status.margin_used = tot_margin
        if tot_margin > 0
            sim_status.margin_level = sim_status.equity / tot_margin
        else
            sim_status.margin_level = 1.
        end
    end
end

function get_week_conv_rates(conv_ratio, datetime::DateTime, bars_in_week::Int)
    week_conv_rate = nothing
    if isa(conv_ratio, DataFrame)
        week_conv_rate = filter(x -> x.datetime >= datetime, conv_ratio)
        if nrow(week_conv_rate) >= bars_in_week
            week_conv_rate = week_conv_rate[1: bars_in_week, :open]
        else
            week_conv_rate = [week_conv_rate[1: end, :open]; [week_conv_rate[end, :open] for _ in 1:bars_in_week-nrow(week_conv_rate)]]
        end
    else
        week_conv_rate = fill(conv_ratio, bars_in_week)
    end
    week_conv_rate
end

function breakeven!(sim_status::SimulatorStatus, bid::Float64, ask::Float64)
    for pos in sim_status.positions
        if pos.dir == "buy"
            if bid > pos.entry_price
                pos.sl_price = pos.entry_price
            end
        else
            if ask < pos.entry_price
                pos.sl_price = pos.entry_price
            end
        end
    end
end

function backtest!(simulation::Simulation; force_init_balance::Bool=true, analysis=false)
    max_balance = 0
    max_drawdown = 0
    acc_quote_rate, acc_base_rate, usd_base_rate, acc_usd_rate = 0., 0., 0., 0.
    conv_rate_i = 0
    last_conv_rate_dt, last_week = 0, 0
    week_acc_quote_rate, week_acc_base_rate, week_usd_base_rate, week_acc_usd_rate = Float64[], Float64[], Float64[], Float64[]
    strategy = simulation.strategy
    # analysis
    signal_logs  = NamedTuple[]
    conv_rates_logs = NamedTuple[]

    for i in 1:nrow(simulation.data)
        datetime = simulation.data[i, :datetime]
        bid = simulation.data[i, :price]
        spread = simulation.sim_params.spread[simulation.symbol] * simulation.sim_params.pip_size
        ask = bid + spread
        last_bar = (open=simulation.data[i, :open], high=simulation.data[i, :high], low=simulation.data[i, :low], close=simulation.data[i, :close])
        simulation.sim_status.datetime = datetime
        simulation.sim_status.bid = bid
        simulation.sim_status.ask = ask
        simulation.sim_status.last_bar = last_bar
        signal = simulation.data[i, :signal]

        # Hack to reduce numbery of querys for conversion rates
        if  (i > 1) && (((dayofweek(datetime) == 7) && (day(datetime) != day(last_conv_rate_dt))) || ((dayofweek(datetime) != 7) && (week(datetime) != last_week)))
            conv_rate_i = 0
        end
        if conv_rate_i == 0
            bars_in_week = length(filter(x -> (x >= datetime - Day(1)) && (x <= datetime + Day(6)), simulation.data.datetime))

            week_acc_quote_rate = get_week_conv_rates(simulation.sim_params.acc_quote_ratio, datetime, bars_in_week)
            week_acc_base_rate = get_week_conv_rates(simulation.sim_params.acc_base_ratio, datetime, bars_in_week)
            week_usd_base_rate = get_week_conv_rates(simulation.sim_params.usd_base_ratio, datetime, bars_in_week)
            week_acc_usd_rate = get_week_conv_rates(simulation.sim_params.acc_usd_ratio, datetime, bars_in_week)

            last_conv_rate_dt = datetime
            if dayofweek(datetime) == 7
                last_week = week(datetime) + 1
            else
                last_week = week(datetime)
            end
            conv_rate_i = 1
        end
        acc_quote_rate = week_acc_quote_rate[conv_rate_i]
        acc_base_rate = week_acc_base_rate[conv_rate_i]
        usd_base_rate = week_usd_base_rate[conv_rate_i]
        acc_usd_rate = week_acc_usd_rate[conv_rate_i]

        update_positions!(simulation.sim_status, 
                          acc_quote_rate, usd_base_rate, acc_usd_rate, spread,
                          simulation.sim_params.commission, simulation.trade_logs)
        if signal != 1 
            # analysis
            if analysis
                push!(signal_logs, (datetime=datetime, signal=signal))
                push!(conv_rates_logs, (datetime=datetime, acc_quote=acc_quote_rate, acc_base=acc_base_rate, usd_base=usd_base_rate, acc_usd=acc_usd_rate))
            end

            if (signal != 4) && (signal != 5)
                if dayofweek(simulation.sim_status.datetime) <= 5

                    dir = (signal == 2 ? "buy" : "sell") # if signal in opposite direction close positon
                    if dir == "buy"
                        for pos in simulation.sim_status.positions
                            if pos.dir == "sell"
                                close_all_pos!(simulation.sim_status, 
                                                acc_quote_rate, usd_base_rate, acc_usd_rate,
                                                simulation.sim_params.commission, simulation.trade_logs)
                                break
                            end
                        end
                    else
                        for pos in simulation.sim_status.positions
                            if pos.dir == "buy"
                                close_all_pos!(simulation.sim_status, 
                                                acc_quote_rate, usd_base_rate, acc_usd_rate,
                                                simulation.sim_params.commission, simulation.trade_logs)
                                break
                            end
                        end
                    end

                    if (dayofweek(simulation.sim_status.datetime) != 5) || (hour(simulation.sim_status.datetime) <= 15)

                        if force_init_balance
                            volume = simulation.sim_params.init_balance * simulation.sim_params.risk * acc_quote_rate / (strategy.params["sl"] * simulation.sim_params.pip_size)
                        else
                            volume = simulation.sim_status.balance * simulation.sim_params.risk * acc_quote_rate / (strategy.params["sl"] * simulation.sim_params.pip_size)
                        end
                        
                        volume = floor(Int, volume / 2000) * 1000 * 2
                        if volume >= 2000
                            margin = volume / simulation.sim_params.leverage[simulation.symbol] / acc_base_rate
                            free_margin = simulation.sim_status.equity - simulation.sim_status.margin_used
                            if simulation.sim_status.margin_level != 1. # if 1 old margin is = 0
                                new_margin_level = simulation.sim_status.equity / ((simulation.sim_status.equity / simulation.sim_status.margin_level) + margin)
                            else
                                if margin > 0
                                    new_margin_level = simulation.sim_status.equity / margin
                                else
                                    new_margin_level = 1.
                                end
                            end
                            if (free_margin > (margin * 1.1)) && (new_margin_level > 0.6) # new margin level if position is opened 
                                limit_time = datetime + Minute(floor(Int, strategy.params["time_limit"]))
                                dir = (signal == 2 ? "buy" : "sell")
                                if dir == "buy"
                                    entry_price = ask
                                    sl_price = ask - (strategy.params["sl"] * simulation.sim_params.pip_size)
                                    tp1_price = ask + (strategy.params["tp_1"] * simulation.sim_params.pip_size)
                                    tp2_price = ask + (strategy.params["tp_2"] * simulation.sim_params.pip_size)
                                else
                                    entry_price = bid
                                    sl_price = bid + (strategy.params["sl"] * simulation.sim_params.pip_size)
                                    tp1_price = bid - (strategy.params["tp_1"] * simulation.sim_params.pip_size)
                                    tp2_price = bid - (strategy.params["tp_2"] * simulation.sim_params.pip_size)
                                end
                                if length(simulation.sim_status.positions) == 0
                                    pos_1 = Position(volume / 2, dir, entry_price, sl_price, tp1_price, datetime, limit_time, margin / 2)
                                    pos_2 = Position(volume / 2, dir, entry_price, sl_price, tp2_price, datetime, limit_time, margin / 2)
                                    push!(simulation.sim_status.positions, pos_1)
                                    push!(simulation.sim_status.positions, pos_2)
                                end
                            end
                        end
                    end
                end
            else
                if signal == 4
                    close_all_pos!(simulation.sim_status, 
                                   acc_quote_rate, usd_base_rate, acc_usd_rate,
                                   simulation.sim_params.commission, simulation.trade_logs)
                elseif signal == 5
                    breakeven!(simulation.sim_status, bid, ask)
                end
            end
        end
        acc_log = (datetime=datetime, balance=simulation.sim_status.balance, equity=simulation.sim_status.equity)
        push!(simulation.acc_logs, acc_log)
        max_balance = max(max_balance, simulation.sim_status.balance)
        drawdown = 1 - (simulation.sim_status.balance / max_balance)
        max_drawdown = max(max_drawdown, drawdown)
        simulation.max_drawdown = max_drawdown
        conv_rate_i += 1
    end
    close_all_pos!(simulation.sim_status, 
                   acc_quote_rate, usd_base_rate, acc_usd_rate,
                   simulation.sim_params.commission, simulation.trade_logs)
    
    (analysis) && (return (signal_logs=signal_logs, conv_rates_logs=conv_rates_logs)) 
end