#!/usr/bin/ruby

def write_two_files(lines)
  if lines.size != 2
    raise "not two lines"
  end

  2.times{|i|
    c = (?a.ord + i).chr
    File.open("/tmp/#{c}", 'w') do |ofile|
      lines[i].split("\n").each_with_index do |line, i|
        ofile.puts line.split(/[ \t]/) * "\n"
        ofile.puts "=== #{i} ===" if lines[i].size > 1
      end
    end
  }
end

if ARGV[1]
  write_two_files([File.read(ARGV[0]), File.read(ARGV[1])])
else
  lines = $<.read.chomp.split("\n")
  write_two_files(lines)
end

exec("diff -u /tmp/a /tmp/b")
