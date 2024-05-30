using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;
using cAlgo.Indicators;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class HistDataExporter : Robot
    {
        [Parameter("Data directory", DefaultValue = "Z:\\Data\\AlgoTrading\\hist_data")]
        public string DataDir { get; set; }

        [Parameter("Buffer size", DefaultValue = 4096)]
        public int BufferSize { get; set; }

        private string[] AllSymbols = new string[] 
        {
            "EURUSD",
            "GBPUSD",
            "USDCHF",
            "USDCAD",
            "AUDUSD",
            "NZDUSD",
            "EURGBP",
            "EURCHF",
            "GBPCHF",
            "AUDCAD",
            "EURCAD",
            "AUDNZD",
            "USDJPY",
            "EURJPY",
            "GBPJPY",
            "CHFJPY",
            "AUDJPY",
            "EURAUD",
            "EURNZD"
        };
        private TimeFrame[] AllTimeFrames = new TimeFrame[] 
        {
            TimeFrame.Minute,
            TimeFrame.Minute5,
            TimeFrame.Minute15,
            TimeFrame.Hour,
            TimeFrame.Hour4,
            TimeFrame.Hour12,
            TimeFrame.Daily
        };

        private Dictionary<string, Bars> AllBars = new Dictionary<string, Bars>();
        private Dictionary<string, BookMark> AllBookMarks = new Dictionary<string, BookMark>();

        System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();

        protected override void OnStart()
        {
            customCulture.NumberFormat.NumberDecimalSeparator = ".";
            System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;

            for (int symb_id = 0; symb_id < AllSymbols.Length; symb_id++)
            {
                for (int tf_id = 0; tf_id < AllTimeFrames.Length; tf_id++)
                {
                    string key = AllSymbols[symb_id] + "-" + AllTimeFrames[tf_id].ToString();
                    AllBars.Add(key, MarketData.GetBars(AllTimeFrames[tf_id], AllSymbols[symb_id]));
                    string dir_path = String.Format("{0}\\{1}\\{2}", DataDir, AllSymbols[symb_id], AllTimeFrames[tf_id].ToString());
                    Directory.CreateDirectory(dir_path);
                    string[] all_files = Directory.GetFiles(dir_path);
                    List<int> last_years = new List<int>();
                    for (int i = 0; i < all_files.Length; i++)
                    {
                        last_years.Add(Int32.Parse(all_files[i].Substring(all_files[i].LastIndexOf('\\') + 1).Replace(".csv", "")));
                    }
                    DateTime last_dt;
                    int curr_year;
                    if (last_years.Count > 0)
                    {
                        string last_file = String.Format("{0}\\{1}.csv", dir_path, last_years.Max());
                        string[] last_line = File.ReadLines(last_file).Last().Trim('\n').Split(',');
                        last_dt = DateTime.ParseExact(last_line[0] + "-" + last_line[1], "yyyy.MM.dd-HH:mm", System.Globalization.CultureInfo.InvariantCulture);
                        curr_year = last_years.Max();
                    }
                    else
                    {
                        last_dt = DateTime.UtcNow;
                        curr_year = 0;
                    }
                    string file_path = dir_path + "\\" + curr_year.ToString() + ".csv";
                    BookMark bookmark = new BookMark(file_path, dir_path, curr_year, last_dt);
                    AllBookMarks.Add(key, bookmark);
                }
            }
        }

        protected override void OnBar()
        {
            for (int symb_id = 0; symb_id < AllSymbols.Length; symb_id++)
            {
                for (int tf_id = 0; tf_id < AllTimeFrames.Length; tf_id++)
                {
                    string key = AllSymbols[symb_id] + "-" + AllTimeFrames[tf_id].ToString();
                    Bar last_bar = AllBars[key].Last(1);
                    BookMark bookmark = AllBookMarks[key];
                    DateTime curr_dt = last_bar.OpenTime;
                    if ((curr_dt > bookmark.LastDt) || (bookmark.CurrYear == 0))
                    {
                        if (bookmark.CurrYear == 0)
                        {
                            bookmark.CurrYear = last_bar.OpenTime.Year;
                            bookmark.LastDt = AllBars[key].Last(2).OpenTime;
                            bookmark.FilePath = bookmark.DirPath + "\\" + bookmark.CurrYear.ToString() + ".csv";
                            File.CreateText(bookmark.FilePath).Close();
                        }
                        if (curr_dt > bookmark.LastDt)
                        {
                            string buff = last_bar.OpenTime.ToString("yyyy.MM.dd") + "," + last_bar.OpenTime.ToString("HH:mm") + "," + last_bar.Open.ToString() + "," + last_bar.High.ToString() + "," + last_bar.Low.ToString() + "," + last_bar.Close.ToString() + "," + last_bar.TickVolume.ToString() + "\n";
                            if ((bookmark.BufferLen < BufferSize) && (last_bar.OpenTime.Year == bookmark.CurrYear))
                            {
                                bookmark.Buffer += buff;
                                bookmark.BufferLen += 1;
                            }
                            else
                            {
                                File.AppendAllText(bookmark.FilePath, bookmark.Buffer);
                                if (last_bar.OpenTime.Year != bookmark.CurrYear)
                                {
                                    bookmark.CurrYear = last_bar.OpenTime.Year;
                                    bookmark.FilePath = DataDir + "\\" + AllSymbols[symb_id] + "\\" + AllTimeFrames[tf_id].ToString() + "\\" + bookmark.CurrYear.ToString() + ".csv";
                                    File.CreateText(bookmark.FilePath).Close();
                                }
                                bookmark.Buffer = buff;
                                bookmark.BufferLen = 1;
                            }
                            bookmark.LastDt = curr_dt;
                        }
                    }
                }
            }
        }

        protected override void OnStop()
        {
            for (int symb_id = 0; symb_id < AllSymbols.Length; symb_id++)
            {
                for (int tf_id = 0; tf_id < AllTimeFrames.Length; tf_id++)
                {
                    string key = AllSymbols[symb_id] + "-" + AllTimeFrames[tf_id].ToString();
                    BookMark bookmark = AllBookMarks[key];
                    if (bookmark.BufferLen > 0)
                    {
                        File.AppendAllText(bookmark.FilePath, bookmark.Buffer);
                    }
                }
            }
        }
    }

    public class BookMark
    {
        public string FilePath { get; set; }
        public string DirPath { get; set; }
        public string Buffer;
        public int BufferLen;
        public int CurrYear { get; set; }
        public DateTime LastDt { get; set; }
        public List<int> LastYears;
        public BookMark(string file_path, string dir_path, int curr_year, DateTime last_dt)
        {
            FilePath = file_path;
            DirPath = dir_path;
            Buffer = "";
            BufferLen = 0;
            CurrYear = curr_year;
            LastDt = last_dt;
            LastYears = new List<int>();
        }
    }
}
