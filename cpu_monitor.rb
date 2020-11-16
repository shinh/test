#!/usr/bin/env ruby

def get
  now = Time.now
  line = File.readlines('/proc/stat')[0]
  metrics = line.split[1, 10].map do |v|
    # USER_HZ = 100
    v.to_i * 0.01
  end
  [now, metrics]
end

metric_names = %w(user nice system idle iowait irq softirq steal guest guest_nice TOTAL)

prev_time, prev_metrics = get
loop do
  sleep 1
  time, metrics = get

  elapsed = time - prev_time
  incrs = metrics.zip(prev_metrics).map{|m, pm|m - pm}
  incrs << incrs.sum

  metric_names.zip(incrs).each do |mn, incr|
    puts "#{mn} #{incr / elapsed}"
  end

  prev_time = time
  prev_metrics = metrics
end

