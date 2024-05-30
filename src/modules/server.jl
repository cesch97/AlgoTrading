
include("strategy.jl")
include("trad_system.jl")
using Sockets
using DataFrames
using Dates


function read_request(sock)
    str_response = strip(readline(sock), '\n')
    if occursin(";", str_response)
        response = split(str_response, ';')
    else
        response = [str_response]
    end
    response
end

function get_data_row(request)
    datetime = DateTime(request[end], DateFormat("y/m/d-H:M"))
    bar_data = parse.([Float64,], replace.(request[4:7], ["," => ".",]))
    (datetime=datetime, open=bar_data[1], high=bar_data[2], low=bar_data[3], close=bar_data[4])
end

function get_allcs(trad_sys::TradSystem, raw_data::OrderedDict)

    used_indicators = Int[] # avoid calculation of unused indicators
    for sel in trad_sys.strat_data.selectors
        ind = sel.indicator
        if !(ind in used_indicators)
            push!(used_indicators, ind)
        end
    end

    ind_data = OrderedDict()
    j = 0
    for (i, ind) in enumerate(trad_sys.strat_data.indicators)
        if i in used_indicators
            ind_id = get_ind_id(ind)
            if !haskey(ind_data, ind_id)
                ind_data[ind_id] = calc_indicator(ind.name, ind.params, raw_data[ind.symbol])
            else
                j += 1
                ind_data[ind_id * "#" * string(j)] = copy(ind_data[ind_id])
            end
        else
            ind_data[string(i)] = NaN # placeholder for non-calculated indicators
        end
    end
    sel_data = OrderedDict()
    std_data = OrderedDict()
    i = 0
    for sel in trad_sys.strat_data.selectors
        sel_id = get_sel_id(sel.name, sel.params, sel.indicator)
        if !haskey(sel_data, sel_id)
            ind_id = clean_id(collect(keys(ind_data))[sel.indicator])
            sel_data[sel_id] = calc_selector(sel.name, sel.params, ind_data[ind_id])
            μ, σ = trad_sys.strat_data.std_params[sel_id]
            std_vec = (sel_data[sel_id] .- μ) ./ σ
            std_data[sel_id] = std_vec
        else
            i += 1
            sel_data[sel_id * "#" * string(i)] = copy(sel_data[sel_id])
            std_data[sel_id * "#" * string(i)] = copy(std_data[sel_id])
        end
    end

    # ind_data_vec = Float64[]
    # for key in keys(ind_data)
    #     push!(ind_data_vec, ind_data[key][end])
    # end
    # ind_data_vec = Matrix{Float32}(reshape(ind_data_vec, (1, size(ind_data_vec)...)))

    # sel_data_vec = Float64[]
    # for key in keys(sel_data)
    #     push!(sel_data_vec, sel_data[key][end])
    # end
    # sel_data_vec = Matrix{Float32}(reshape(sel_data_vec, (1, size(sel_data_vec)...)))

    data_vec = Float64[]
    for key in keys(std_data)
        push!(data_vec, std_data[key][end])
    end
    data_vec = Matrix{Float32}(reshape(data_vec, (1, size(data_vec)...)))

    # global serv_ind_data, serv_sel_data, serv_std_data
    # push!(serv_ind_data, tuple(ind_data_vec[1, :]...))
    # push!(serv_sel_data, tuple(sel_data_vec[1, :]...))
    # push!(serv_std_data, tuple(data_vec[1, :]...))

    forward(trad_sys.model, data_vec; arg_max=false)[1, :]
end

# serv_ind_data = Tuple[]
# serv_sel_data = Tuple[]
# serv_std_data = Tuple[]

struct PositionData
    volume::Number
    dir::String
    entry_price::Float64
    entry_time::DateTime
    profit::Float64
end

function get_position(request)
    datetime = DateTime(request[1], DateFormat("y/m/d-H:M"))
    pos_data = parse.([Float64,], replace.(request[3:5], ["," => ".",]))
    PositionData(pos_data[1], request[2], pos_data[2], datetime, pos_data[3])
end

function get_signal(strategy::Strategy, raw_data::DataFrame)

    used_indicators = Int[] # avoid calculation of unused indicators
    for sel in strategy.strat_data.selectors
        ind = sel.indicator
        if !(ind in used_indicators)
            push!(used_indicators, ind)
        end
    end

    ind_data = OrderedDict()
    j = 0
    for (i, ind) in enumerate(strategy.strat_data.indicators)
        if i in used_indicators
            ind_id = get_ind_id(ind)
            if !haskey(ind_data, ind_id)
                ind_data[ind_id] = calc_indicator(ind.name, ind.params, raw_data)
            else
                j += 1
                ind_data[ind_id * "#" * string(j)] = copy(ind_data[ind_id])
            end
        else
            ind_data[string(i)] = NaN # placeholder for non-calculated indicators
        end
    end
    sel_data = OrderedDict()
    std_data = OrderedDict()
    i = 0
    for sel in strategy.strat_data.selectors
        sel_id = get_sel_id(sel.name, sel.params, sel.indicator)
        if !haskey(sel_data, sel_id)
            ind_id = clean_id(collect(keys(ind_data))[sel.indicator])
            sel_data[sel_id] = calc_selector(sel.name, sel.params, ind_data[ind_id])
            μ, σ = strategy.strat_data.std_params[sel_id]
            std_vec = (sel_data[sel_id] .- μ) ./ σ
            std_data[sel_id] = std_vec
        else
            i += 1
            sel_data[sel_id * "#" * string(i)] = copy(sel_data[sel_id])
            std_data[sel_id * "#" * string(i)] = copy(std_data[sel_id])
        end
    end
    data_vec = Float64[]
    for key in keys(std_data)
        push!(data_vec, std_data[key][end])
    end
    data_vec = Matrix{Float32}(reshape(data_vec, (1, size(data_vec)...)))
    forward(strategy.model, data_vec; arg_max=true)[1]
end