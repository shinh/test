require 'rubygems'
require 'pstore'
require 'mechanize'

def parse_time(s)
  if s =~ /(\d\d\d\d)\/(\d\d)\/(\d\d)/
    Time.mktime($1.to_i, $2.to_i, $3.to_i)
  end
end

def time_str(t)
  t.strftime('%Y/%m/%d')
end

def store(dbname, page)
  min_t = nil

  m = {}
  page.at('#tblHistoryTable').children.each do |x|
    a = x.children.map{|_|_.children[0].to_s}
    next if !a[0]
    t = parse_time(a[0])
    next if !t
    if !min_t || min_t > t
      min_t = t
    end
    m[a[0]] = a[1, 4].map(&:to_f)
  end

  db = PStore.new('goog.db')
  db.transaction do
    if !db[dbname]
      db[dbname] = {}
    end
    m.each do |k, v|
      db[dbname][k] = v
    end
  end

  min_t
end

def update_db(dbname, symbol)
  last_t = Time.mktime(2007,1,1)
  db = PStore.new('goog.db')
  db.transaction(true) do
    if db[dbname]
      last_t = parse_time(db[dbname].keys.max)
    end
  end

  STDERR.puts "Fetch #{dbname} until #{time_str(last_t)}..."

  ua = Mechanize.new
  page = ua.get(URI.parse("http://jp.moneycentral.msn.com/investor/charts/historicdata.aspx?symbol=#{symbol}"))
  next_t = store(dbname, page)

  while next_t > last_t
    STDERR.puts "Fetch #{dbname} for #{time_str(next_t)}..."

    form = page.forms[2]
    form.txtStartDate = time_str(next_t - 60*60*24*30)
    form.txtEndDate = time_str(next_t)
    form.rbDateRange = '7'

    page = form.submit
    next_t = store(dbname, page)
  end
end

if !ARGV[0]
  update_db('goog', 'us%3agoog')
  update_db('yen', '/jpyusd')
end

db = PStore.new('goog.db')
goog, yen = db.transaction(true) do
  [db['goog'], db['yen']]
end

values = []
File.open('goog.dat', 'w') do |dat|
  t = Time.mktime(2008,4,1)
  while t < Time.now
    ts = time_str(t)
    if goog[ts] && yen[ts]
      y = goog[ts][0] * yen[ts][0]
      #STDERR.puts "#{y} = #{goog[ts][0]} #{yen[ts][0]}"
      values << [t, y]
      dat.puts "#{ts} #{y}"
    end

    t += 60*60*24
  end
end

system(%q(gnuplot -e '
set terminal png;
set output "goog.png";
set xdata time;
set format x "%y/%m";
set timefmt "%Y/%m/%d";
plot "goog.dat" using 1:2;
'))

File.open('goog.html', 'w') do |html|
  html.puts %Q(<html>
  <head>
    <title>goog in yen</title>
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
var D = [
)

  values.each do |t, y|
    html.puts %Q([#{t.year}, #{t.mon-1}, #{t.day}, #{y}],)
  end

  html.puts %Q(
];
      google.load("visualization", "1", {packages:["corechart"]});
      google.setOnLoadCallback(drawChart);
      function drawChart() {
        document.getElementById('chart_div').innerHTML = "";
        var data = new google.visualization.DataTable();
        data.addColumn('date', 'date');
        data.addColumn('number', 'goog');
        var from = new Date(document.getElementById("year").value,
                            document.getElementById("month").value - 1,
                            document.getElementById("day").value);
        var filtered = [];
        for (var i = 0; i < D.length; i++) {
          var d = new Date(D[i][0], D[i][1], D[i][2]);
          if (d < from) continue;
          filtered.push([d, D[i][3]]);
        }
        data.addRows(filtered.length);
        for (var i = 0; i < filtered.length; i++) {
          data.setValue(i, 0, filtered[i][0]);
          data.setValue(i, 1, filtered[i][1]);
        }
        var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
        chart.draw(data, {width: 640, height: 480, title: 'goog in yen'});
      }
    </script>
  </head>

  <body>
    <div id="chart_div"></div>
    from
     <input id="year" size=4 value=2008>
     <input id="month" size=2 value=4>
     <input id="day" size=2 value=1>
     <input type="button" value="update" onclick="drawChart()">
  </body>
</html>
)
end
