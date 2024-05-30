using System;
using System.Text;
using System.Net.Sockets;
using System.Globalization;
using System.Collections.Generic;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;
using cAlgo.Indicators;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class ClientTradSystem : Robot
    {
        [Parameter(DefaultValue = "192.168.0.111")]
        public string IpAddress { get; set; }

        [Parameter(DefaultValue = 8080)]
        public int PortNum { get; set; }

        private int BuffSize = 1024;
        // Buffer for reciving server response in bytes
        private string strategy_name;
        private TcpClient ClientTcp;
        private NetworkStream ns;
        private string request;
        private string[] response;
        int num_old_bars;
        private List<Bars> trad_sys_sources = new List<Bars>();
        private List<Bars> strats_sources = new List<Bars>();
        private List<int> trad_sys_counts = new List<int>();
        private List<int> strats_counts = new List<int>();
        private List<ConvRate> acc_quote_rates = new List<ConvRate>();
        private List<ConvRate> acc_base_rates = new List<ConvRate>();


        protected override void OnStart()
        {
            // setting up the client
            ClientTcp = new TcpClient(IpAddress, PortNum);
            ns = ClientTcp.GetStream();
            WriteRequest("Connected to Julia Client!");
            response = ReadResponse();
            strategy_name = response[0];
            WriteRequest(String.Format("{0};{1};{2}", strategy_name, Account.Balance, Account.Number));

            // setting up the data sources
            while (true)
            {
                response = ReadResponse();
                if (response[0] == "0")
                {
                    break;
                }
                Bars bars = MarketData.GetBars(ParseTimeFrame(response[1]), response[0]);
                trad_sys_sources.Add(bars);
                trad_sys_counts.Add(bars.Count);
                WriteRequest("0");
            }
            while (true)
            {
                response = ReadResponse();
                if (response[0] == "0")
                {
                    break;
                }
                Bars bars = MarketData.GetBars(ParseTimeFrame(response[1]), response[0]);
                strats_sources.Add(bars);
                strats_counts.Add(bars.Count);
                // acc_quote_rate
                string acc_quote = String.Format("{0}{1}", "EUR", response[0].Substring(3, 3));
                ConvRate acc_quote_rate;
                if (Symbols.Exists(acc_quote))
                {
                    Bars conv_bars = MarketData.GetBars(ParseTimeFrame(response[1]), acc_quote);
                    acc_quote_rate = new ConvRate(conv_bars, false, false);
                }
                else
                {
                    acc_quote = String.Format("{0}{1}", response[0].Substring(3, 3), "EUR");
                    if (Symbols.Exists(acc_quote))
                    {
                        Bars conv_bars = MarketData.GetBars(ParseTimeFrame(response[1]), acc_quote);
                        acc_quote_rate = new ConvRate(conv_bars, true, false);
                    }
                    else
                    {
                        acc_quote_rate = new ConvRate(bars, false, true);
                    }
                }
                acc_quote_rates.Add(acc_quote_rate);
                // acc_base_rate
                string acc_base = String.Format("{0}{1}", "EUR", response[0].Substring(0, 3));
                ConvRate acc_base_rate;
                if (Symbols.Exists(acc_base))
                {
                    Bars conv_bars = MarketData.GetBars(ParseTimeFrame(response[1]), acc_base);
                    acc_base_rate = new ConvRate(conv_bars, false, false);
                }
                else
                {
                    acc_base = String.Format("{0}{1}", response[0].Substring(0, 3), "EUR");
                    if (Symbols.Exists(acc_base))
                    {
                        Bars conv_bars = MarketData.GetBars(ParseTimeFrame(response[1]), acc_base);
                        acc_base_rate = new ConvRate(conv_bars, true, false);
                    }
                    else
                    {
                        acc_base_rate = new ConvRate(bars, false, true);
                    }
                }
                acc_base_rates.Add(acc_base_rate);
                WriteRequest("0");
            }

            // retrieving old data (only for live trading!)
            response = ReadResponse();
            num_old_bars = Int32.Parse(response[0]);
            if (IsBacktesting)
            {
                WriteRequest("0");
                response = ReadResponse();
            }
            else
            {
                for (int i = 0; i < trad_sys_sources.Count; i++)
                {
                    Bars bars = trad_sys_sources[i];
                    while (bars.Count < num_old_bars)
                    {
                        bars.LoadMoreHistory();
                        Print(String.Format("{0} - {1}", bars.SymbolName, bars.Count));
                    }
                    for (int j = num_old_bars + 50; j > 0; j--) // it should start compute allcs 2-3 weeks early
                    {
                        DateTime datetime = bars.OpenTimes.Last(j - 1);
                        DateTime bar_datetime = bars.OpenTimes.Last(j);
                        double open = bars.OpenPrices.Last(j);
                        double high = bars.HighPrices.Last(j);
                        double low = bars.LowPrices.Last(j);
                        double close = bars.ClosePrices.Last(j);
                        request = String.Format("trad_system;{0};{1:yyyy/MM/dd-HH:mm};{2};{3};{4};{5};{6:yyyy/MM/dd-HH:mm}", i, bar_datetime, open, high, low, close, datetime);
                        WriteRequest(request);
                        response = ReadResponse();
                    }
                    WriteRequest("0");
                    response = ReadResponse();
                }
                for (int i = 0; i < strats_sources.Count; i++)
                {
                    Bars bars = strats_sources[i];
                    while (bars.Count < num_old_bars)
                    {
                        bars.LoadMoreHistory();
                        Print(String.Format("{0} - {1}", bars.SymbolName, bars.Count));
                    }
                    for (int j = num_old_bars; j > 0; j--)
                    {
                        DateTime datetime = bars.OpenTimes.Last(j - 1);
                        DateTime bar_datetime = bars.OpenTimes.Last(j);
                        double open = bars.OpenPrices.Last(j);
                        double high = bars.HighPrices.Last(j);
                        double low = bars.LowPrices.Last(j);
                        double close = bars.ClosePrices.Last(j);
                        request = String.Format("strategies;{0};{1:yyyy/MM/dd-HH:mm};{2};{3};{4};{5};{6:yyyy/MM/dd-HH:mm}", i, bar_datetime, open, high, low, close, datetime);
                        WriteRequest(request);
                        response = ReadResponse();
                    }
                    WriteRequest("0");
                    response = ReadResponse();
                }
            }
        }

        protected override void OnBar()
        {
            for (int i = 0; i < trad_sys_sources.Count; i++)
            {
                Bars bars = trad_sys_sources[i];
                if (bars.Count > trad_sys_counts[i])
                {
                    DateTime datetime = bars.OpenTimes.Last(0);
                    DateTime bar_datetime = bars.OpenTimes.Last(1);
                    double open = bars.OpenPrices.Last(1);
                    double high = bars.HighPrices.Last(1);
                    double low = bars.LowPrices.Last(1);
                    double close = bars.ClosePrices.Last(1);
                    trad_sys_counts[i] = bars.Count;
                    double balance = Account.Balance;
                    request = String.Format("trad_system;{0};{1:yyyy/MM/dd-HH:mm};{2};{3};{4};{5};{6};{7:yyyy/MM/dd-HH:mm}", i, bar_datetime, open, high, low, close, balance, datetime);
                    WriteRequest(request);
                    response = ReadResponse();
                }
            }
            for (int i = 0; i < strats_sources.Count; i++)
            {
                Bars bars = strats_sources[i];
                if (bars.Count > strats_counts[i])
                {
                    DateTime datetime = bars.OpenTimes.Last(0);
                    DateTime bar_datetime = bars.OpenTimes.Last(1);
                    double open = bars.OpenPrices.Last(1);
                    double high = bars.HighPrices.Last(1);
                    double low = bars.LowPrices.Last(1);
                    double close = bars.ClosePrices.Last(1);
                    double bid = Symbols.GetSymbol(bars.SymbolName).Bid;
                    double ask = Symbols.GetSymbol(bars.SymbolName).Ask;
                    strats_counts[i] = bars.Count;
                    request = String.Format("strategies;{0};{1:yyyy/MM/dd-HH:mm};{2};{3};{4};{5};{6};{7};{8:yyyy/MM/dd-HH:mm}", i, bar_datetime, open, high, low, close, bid, ask, datetime);
                    WriteRequest(request);
                    response = ReadResponse();
                    if (response[0] != "skip")
                    {
                        // checking if positions expired
                        UpdatePositions(i);
                        response = ReadResponse();
                        if (response[0] != "0")
                        {
                            if (response[0] == "open_pos")
                            {
                                OpenPosition(i, response);
                            }
                            else if (response[0] == "close_all")
                            {
                                string pos_label = String.Format("{0}-{1}", strategy_name, i);
                                Position[] positions = Positions.FindAll(pos_label, strats_sources[i].SymbolName);
                                foreach (Position position in positions)
                                {
                                    position.Close();
                                }
                                WriteRequest("0");
                                response = ReadResponse();
                                if (response[0] == "open_pos")
                                {
                                    OpenPosition(i, response);
                                }
                            }
                            else if (response[0] == "breakeven")
                            {
                                BreakEven(i);
                            }
                        }
                    }
                }
            }
        }

        protected override void OnStop()
        {
            if (IsBacktesting)
            {
                for (int i = 0; i < strats_sources.Count; i++)
                {
                    string pos_label = String.Format("{0}-{1}", strategy_name, i);
                    Position[] positions = Positions.FindAll(pos_label, strats_sources[i].SymbolName);
                    foreach (Position position in positions)
                    {
                        position.Close();
                    }
                }
            }
            WriteRequest("1");
            ClientTcp.Close();
        }

        void WriteRequest(string request)
        {
            byte[] byteRequest = Encoding.Default.GetBytes(String.Format("{0}\n", request));
            ns.Write(byteRequest, 0, byteRequest.Length);
        }

        string[] ReadResponse()
        {
            byte[] byteResponse = new byte[BuffSize];
            ns.Read(byteResponse, 0, byteResponse.Length);
            string str_response = Encoding.Default.GetString(byteResponse).Trim('\n', '\0');
            string[] response;
            if (str_response.Contains(';'))
            {
                response = str_response.Split(';');
            }
            else
            {
                response = new string[] 
                {
                    str_response
                };
            }
            return response;
        }

        TimeFrame ParseTimeFrame(string str_timeframe)
        {
            if (str_timeframe == "Daily")
            {
                return TimeFrame.Daily;
            }
            if (str_timeframe == "Hour12")
            {
                return TimeFrame.Hour12;
            }
            if (str_timeframe == "Hour4")
            {
                return TimeFrame.Hour4;
            }
            if (str_timeframe == "Hour")
            {
                return TimeFrame.Hour;
            }
            if (str_timeframe == "Minute15")
            {
                return TimeFrame.Minute15;
            }
            if (str_timeframe == "Minute5")
            {
                return TimeFrame.Minute5;
            }
            if (str_timeframe == "Minute")
            {
                return TimeFrame.Minute;
            }
            else
            {
                return TimeFrame;
            }
        }

        void UpdatePositions(int i)
        {
            string pos_label = String.Format("{0}-{1}", strategy_name, i);
            Position[] positions = Positions.FindAll(pos_label, strats_sources[i].SymbolName);
            foreach (Position position in positions)
            {
                DateTime entry_time = position.EntryTime;
                double volume = position.VolumeInUnits;
                TradeType pos_type = position.TradeType;
                double entry_price = position.EntryPrice;
                string dir;
                if (pos_type == TradeType.Buy)
                {
                    dir = "buy";
                }
                else
                {
                    dir = "sell";
                }
                double profit = position.NetProfit;
                request = String.Format("{0:yyyy/MM/dd-HH:mm};{1};{2};{3};{4}", entry_time, dir, volume, entry_price, profit);
                WriteRequest(request);
                response = ReadResponse();
                if (response[0] == "close")
                {
                    position.Close();
                }
            }
            WriteRequest("0");
            ReadResponse();
            double balance = Account.Balance;
            double acc_quote_rate = acc_quote_rates[i].GetRate();
            double acc_base_rate = acc_base_rates[i].GetRate();
            double pip_size = Symbols.GetSymbol(strats_sources[i].SymbolName).PipSize;
            WriteRequest(String.Format("{0};{1};{2};{3}", balance, acc_quote_rate, acc_base_rate, pip_size));
        }

        void OpenPosition(int i, string[] response)
        {
            string pos_label = String.Format("{0}-{1}", strategy_name, i);
            var dir = TradeType.Buy;
            if (response[1] == "sell")
            {
                dir = TradeType.Sell;
            }
            double volume = float.Parse(response[2], CultureInfo.InvariantCulture.NumberFormat);
            double sl_pips = int.Parse(response[3]);
            double tp1_pips = int.Parse(response[4]);
            double tp2_pips = int.Parse(response[5]);
            ExecuteMarketOrder(dir, strats_sources[i].SymbolName, volume, pos_label, sl_pips, tp1_pips);
            ExecuteMarketOrder(dir, strats_sources[i].SymbolName, volume, pos_label, sl_pips, tp2_pips);
        }

        void BreakEven(int i)
        {
            string pos_label = String.Format("{0}-{1}", strategy_name, i);
            Position[] positions = Positions.FindAll(pos_label, strats_sources[i].SymbolName);
            double bid = Symbols.GetSymbol(strats_sources[i].SymbolName).Bid;
            double ask = Symbols.GetSymbol(strats_sources[i].SymbolName).Ask;
            foreach (Position position in positions)
            {
                TradeType pos_type = position.TradeType;
                double entry_price = position.EntryPrice;
                if (pos_type == TradeType.Buy)
                {
                    if (bid > entry_price)
                    {
                        position.ModifyStopLossPrice(entry_price);
                    }
                }
                else
                {
                    if (ask < entry_price)
                    {
                        position.ModifyStopLossPrice(entry_price);
                    }
                }
            }
        }
    }

    public class ConvRate
    {
        public Bars bars { get; set; }
        public bool invert { get; set; }
        public bool one { get; set; }
        public ConvRate(Bars _bars, bool _invert, bool _one)
        {
            bars = _bars;
            invert = _invert;
            one = _one;
        }

        public double GetRate()
        {
            if (!one)
            {
                if (!invert)
                {
                    return bars.Last(1).Close;
                }
                return 1 / bars.Last(1).Close;
            }
            else
            {
                return 1.0;
            }
        }
    }
}
