
include("modules/server.jl")
using YAML
using Dates


function trad_sys_serving(config_file)
    if isa(config_file, String)
        config = YAML.load_file(config_file)
        log_dir = config["log_dir"]
    else
        config = config_file
        log_dir = config["trad_logs_dir"]
    end
    port_num = config["port_num"]
    data_dir = config["data_dir"]
    strats_dir = config["strats_dir"]
    trad_sys_dir = config["trad_sys_dir"]
    master_log = config["master_log"]
    trad_sys_name = config["trad_sys_name"]
    start_from = config["start_from"]
    num_old_bars = config["num_old_bars"]
    risk = config["risk"]
    leverage = config["leverage"]
    acc_curr = config["acc_curr"]
    force_init_balance = config["force_init_balance"]

    # loading trad_system and strategies
    trad_sys = load_trad_sys(trad_sys_dir * "/trad_systems/" * trad_sys_name * ".jld2")
    strategies = Strategy[]
    for strat_name in trad_sys.strategies
        strategy = load_strategy(strats_dir * "/strategies/" * strat_name * ".jld2")
        push!(strategies, strategy)
    end
    master_log_file = "$log_dir/$master_log.csv"
    ts_log_file = "$log_dir/$trad_sys_name.csv"

    # setting up the server
    server = listen(IPv4(0), port_num)
    sock = accept(server)
    request = read_request(sock)
    println(request[1])
    response = "$trad_sys_name\n"
    write(sock, response)
    request = read_request(sock)
    println("running: $(request[1])")
    # retrieving the balance
    init_balance = parse(Float64, replace(request[2], "," => ".")) 
    acc_balance = init_balance
    acc_number = parse(Int, request[3]) 

    # setting up data sources
    ts_raw_data = OrderedDict()
    preload_ts_raw_data = OrderedDict()
    for ind in trad_sys.strat_data.indicators
        if !haskey(ts_raw_data, ind.symbol)
            ts_raw_data[ind.symbol] = DataFrame()
            preload_ts_raw_data[ind.symbol] = DataFrame()
            write(sock, "$(ind.symbol);Hour12\n")
            read_request(sock)
        end
    end
    ts_raw_keys = collect(keys(ts_raw_data))
    write(sock, "0\n")
    strats_raw_data = DataFrame[]
    for strat in strategies
        push!(strats_raw_data, DataFrame())
        write(sock, "$(strat.symbol);$(strat.time_frame)\n")
        read_request(sock)
    end
    write(sock, "0\n")

    last_datetime = nothing
    last_week = nothing

    # retrieving old data (only for live trading!)
    write(sock, "$num_old_bars\n")
    request = read_request(sock)
    allcs = zeros(length(strategies) + 1)
    if request[1] == "0"
        write(sock, "0\n")
    else
        for i in eachindex(ts_raw_keys) # push more than the num_old_bars into a temporary struct
            (i != 1) && (request = read_request(sock))
            while true
                if request[1] == "0"
                    write(sock, "0\n")
                    break
                end
                row = get_data_row(request)
                push!(preload_ts_raw_data[ts_raw_keys[i]], row) #
                write(sock, "0\n")
                request = read_request(sock)
            end
        end

        for datetime in preload_ts_raw_data[ts_raw_keys[length(ts_raw_keys)]].datetime
            # simulate the OnBar() method to calculate the allocations
            enough_data = true
            for i in eachindex(ts_raw_keys)
                ts_raw_data[ts_raw_keys[i]] = filter(x -> x.datetime <= datetime, preload_ts_raw_data[ts_raw_keys[i]])
                if nrow(ts_raw_data[ts_raw_keys[i]]) < num_old_bars
                    enough_data = false
                end
            end
            if enough_data
                i = length(ts_raw_keys)
                if isnothing(last_datetime)
                    if week(ts_raw_data[ts_raw_keys[i]][end, :datetime]) != week(ts_raw_data[ts_raw_keys[i]][end-1, :datetime])
                        allcs = get_allcs(trad_sys, ts_raw_data)
                        last_datetime = ts_raw_data[ts_raw_keys[i]][end, :datetime]
                        last_week = week(last_datetime)
                    elseif  day(ts_raw_data[ts_raw_keys[i]][end, :datetime]) == 7
                        allcs = get_allcs(trad_sys, ts_raw_data)               
                        last_datetime = ts_raw_data[ts_raw_keys[i]][end, :datetime]
                        last_week = week(last_datetime) + 1
                    end
                else
                    if (week(ts_raw_data[ts_raw_keys[i]][end, :datetime]) != last_week) && (day(ts_raw_data[ts_raw_keys[i]][end, :datetime]) != day(last_datetime))
                        allcs = get_allcs(trad_sys, ts_raw_data)
                        last_datetime = ts_raw_data[ts_raw_keys[i]][end, :datetime]
                        last_week = week(last_datetime)
                    elseif  (dayofweek(ts_raw_data[ts_raw_keys[i]][end, :datetime]) == 7) && (day(ts_raw_data[ts_raw_keys[i]][end, :datetime]) != day(last_datetime))
                        allcs = get_allcs(trad_sys, ts_raw_data)
                        last_datetime = ts_raw_data[ts_raw_keys[i]][end, :datetime]
                        last_week = week(last_datetime) + 1
                    end
                end
            end
        end
        for i in eachindex(ts_raw_keys)
            ts_raw_data[ts_raw_keys[i]] = last(preload_ts_raw_data[ts_raw_keys[i]], num_old_bars)
        end

        for i in eachindex(strats_raw_data)
            request = read_request(sock)
            while true
                if request[1] == "0"
                    write(sock, "0\n")
                    break
                end
                row = get_data_row(request)
                push!(strats_raw_data[i], row)
                if nrow(strats_raw_data[i]) > num_old_bars
                    strats_raw_data[i] = strats_raw_data[i][2: nrow(strats_raw_data[i]), :]
                end
                write(sock, "0\n")
                request = read_request(sock)
            end
        end
    end

    # analysis
    # signal_logs = NamedTuple[]
    # allc_logs = Tuple[]

    while isopen(sock)
        request = read_request(sock) # stop signal
        if request[1] == "1"
            break
        end
        datetime = DateTime(request[end], DateFormat("y/m/d-H:M")) 
        row = get_data_row(request)
        i = parse(Int, request[2]) + 1
        if request[1] == "trad_system"            
            push!(ts_raw_data[ts_raw_keys[i]], row)
            if nrow(ts_raw_data[ts_raw_keys[i]]) > num_old_bars
                ts_raw_data[ts_raw_keys[i]] = ts_raw_data[ts_raw_keys[i]][2: end, :]
                if (i == length(ts_raw_keys)) 
                    if isnothing(last_datetime)
                        if week(ts_raw_data[ts_raw_keys[i]][end, :datetime]) != week(ts_raw_data[ts_raw_keys[i]][end-1, :datetime])
                            allcs = get_allcs(trad_sys, ts_raw_data)
                            acc_balance = parse(Float64, replace(request[8], "," => ".")) 
                            last_datetime = ts_raw_data[ts_raw_keys[i]][end, :datetime]
                            last_week = week(last_datetime)
                        elseif  day(ts_raw_data[ts_raw_keys[i]][end, :datetime]) == 7
                            allcs = get_allcs(trad_sys, ts_raw_data)
                            acc_balance = parse(Float64, replace(request[8], "," => "."))                    
                            last_datetime = ts_raw_data[ts_raw_keys[i]][end, :datetime]
                            last_week = week(last_datetime) + 1
                        end
                    else
                        if (week(ts_raw_data[ts_raw_keys[i]][end, :datetime]) != last_week) && (day(ts_raw_data[ts_raw_keys[i]][end, :datetime]) != day(last_datetime))
                            allcs = get_allcs(trad_sys, ts_raw_data)
                            acc_balance = parse(Float64, replace(request[8], "," => ".")) 
                            last_datetime = ts_raw_data[ts_raw_keys[i]][end, :datetime]
                            last_week = week(last_datetime)

                            # allc_log = (datetime, allcs...)
                            # push!(allc_logs, allc_log)

                        elseif  (dayofweek(ts_raw_data[ts_raw_keys[i]][end, :datetime]) == 7) && (day(ts_raw_data[ts_raw_keys[i]][end, :datetime]) != day(last_datetime))
                            allcs = get_allcs(trad_sys, ts_raw_data)
                            acc_balance = parse(Float64, replace(request[8], "," => ".")) 
                            last_datetime = ts_raw_data[ts_raw_keys[i]][end, :datetime]
                            last_week = week(last_datetime) + 1

                            # allc_log = (datetime, allcs...)
                            # push!(allc_logs, allc_log)

                        end
                    end
                end
            end
            write(sock, "0\n")
        elseif request[1] == "strategies"
            push!(strats_raw_data[i], row)
            if nrow(strats_raw_data[i]) > num_old_bars
                strats_raw_data[i] = strats_raw_data[i][2: nrow(strats_raw_data[i]), :]
            end
            if ((!isnothing(start_from) && (datetime < start_from)) || (nrow(strats_raw_data[i]) < num_old_bars))
                write(sock, "skip\n")
                continue
            end
            if allcs == zeros(length(strategies) + 1)
                write(sock, "skip\n")
                continue
            end
            bid = parse(Float64, replace(request[7], "," => ".")) # current market price
            ask = parse(Float64, replace(request[8], "," => "."))
            write(sock, "0\n")
            strategy = strategies[i]
            positions = PositionData[]
            tot_margin = 0
            tot_profit = 0
            while true # checking if positions expired
                request = read_request(sock)
                if request[1] == "0"
                    write(sock, "0\n")
                    break
                end
                position = get_position(request)
                if  datetime >= position.entry_time + Minute(strategy.params["time_limit"])            
                    write(sock, "close\n")
                elseif (dayofweek(datetime) == 5) && (hour(datetime) >= 17) # needs to be the same as the simulator
                    write(sock, "close\n")
                else
                    tot_margin += position.volume / leverage[strategy.symbol]
                    tot_profit += position.profit
                    push!(positions, position)
                    write(sock, "0\n")
                end 
            end
            request = read_request(sock)
            curr_balance = parse(Float64, replace(request[1], "," => "."))

            if isnothing(start_from)
                open(master_log_file, "a") do io
                    str_dt = Dates.format(datetime, "yyyy-mm-dd HH:MM")
                    write(io, "$str_dt;$acc_number;$trad_sys_name;$curr_balance\n")
                end
                open(ts_log_file, "a") do io
                    str_dt = Dates.format(datetime, "yyyy-mm-dd HH:MM")
                    write(io, "$str_dt;$acc_number;$curr_balance\n")
                end
            end

            # not exactly like the simulation...
            if force_init_balance
                balance = init_balance * allcs[i+1]
            else
                balance = acc_balance * allcs[i+1] # balance allocated for the strategy
            end

            acc_quote_rate = parse(Float64, replace(request[2], "," => "."))
            acc_base_rate = parse(Float64, replace(request[3], "," => "."))
            tot_margin /= acc_base_rate
            equity = balance + tot_profit
            free_margin = equity - tot_margin
            if tot_margin > 0
                margin_level = equity / tot_margin
            else
                margin_level = 1.
            end
            pip_size = parse(Float64, replace(request[4], "," => "."))
            signal = get_signal(strategy, strats_raw_data[i])

            response = "0\n"
            if signal != 1

                # push!(signal_logs, (strategy=i, datetime=datetime, signal=signal))

                if (signal != 4) && (signal != 5)
                    if dayofweek(datetime) <= 5

                        dir = (signal == 2 ? "buy" : "sell")  
                        if dir == "buy"
                            for pos in positions
                                if pos.dir == "sell"
                                    write(sock, "close_all\n")
                                    read_request(sock)
                                    positions = PositionData[]
                                    balance = equity
                                    equity = balance
                                    free_margin = balance
                                    margin_level = 1.
                                    break
                                end
                            end
                        elseif dir == "sell"
                            for pos in positions
                                if pos.dir == "buy"
                                    write(sock, "close_all\n")
                                    read_request(sock)
                                    positions = PositionData[]
                                    balance = equity
                                    equity = balance
                                    free_margin = balance
                                    margin_level = 1.
                                    break
                                end
                            end
                        end

                        if (dayofweek(datetime) != 5) || (hour(datetime) <= 15)
                            volume = (balance * risk * acc_quote_rate) / (strategy.params["sl"] * pip_size)
                            volume = floor(Int, volume / 2000) * 1000 * 2     
                            margin = volume / leverage[strategy.symbol] / acc_base_rate 
                            if margin_level != 1.
                                new_marg_lev = equity / ((equity / margin_level) + margin)
                            else
                                if margin > 0
                                    new_marg_lev = equity / margin
                                else
                                    new_marg_lev = 1.
                                end
                            end

                            if (volume >= 2000) && (free_margin > (margin * 1.1)) && (new_marg_lev > 0.6) && length(positions) == 0
                                response = "open_pos;$dir;$(volume / 2);$(strategy.params["sl"]);$(strategy.params["tp_1"]);$(strategy.params["tp_2"])\n"                                   
                            end
                        end
                    end
                    write(sock, response)
                else
                    if signal == 4
                        write(sock, "close_all\n")
                        read_request(sock)
                        write(sock, "0\n")
                    elseif signal == 5
                        write(sock, "breakeven\n")
                    end
                end
            else
                write(sock, "0\n")
            end
        end
    end
    # stopping
    close(sock)
    close(server)

    # global serv_signal_logs, serv_allc_logs
    # serv_signal_logs = sort(DataFrame(signal_logs), [:datetime,])
    # serv_allc_logs = DataFrame(allc_logs)
end


# serv_signal_logs = nothing
# serv_allc_logs = nothing
