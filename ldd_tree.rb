#!/usr/bin/ruby

if !ARGV[0]
  STDERR.puts "Usage: #$0 binary"
  exit
end

def show_tree(binary, seen, runpaths, syspaths, nest)
  dynamic = `readelf -d #{binary}`
  dynamic.scan(/\(RUNPATH\).*?: \[(.*)\]/) do
    runpaths += $1.split(':')
  end

  indent = "  " * nest

  dynamic.scan(/\(NEEDED\).*?: \[(.*)\]/) do
    needed = $1
    if seen[needed]
      puts indent + "(#{needed})"
    else
      seen[needed] = true
      resolved = nil
      (runpaths + syspaths).each do |rp|
        r = "#{rp}/#{needed}"
        if File.exist?(r)
          resolved = r
          break
        end
      end

      if resolved
        puts indent + resolved
        show_tree(resolved, seen, runpaths, syspaths, nest + 1)
      else
        puts indent + "#{needed}? (not found)"
      end
    end
  end

end

runpaths = []
if ENV['LD_LIBRARY_PATH']
  runpaths = ENV['LD_LIBRARY_PATH'].split(':')
end

syspaths = []
Dir.glob('/etc/ld.so.conf.d/*.conf').each do |conf|
  File.readlines(conf).each do |line|
    next if /^#/ =~ line
    syspaths << line.chomp
  end
end


puts ARGV[0]
show_tree(ARGV[0], {}, runpaths, syspaths, 1)
