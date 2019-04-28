#!/usr/bin/env ruby
#
# for ChainerCV
#

require 'fileutils'

Dir.chdir(ARGV[0])

def make(kind)
  FileUtils.mkdir_p kind
  File.readlines("#{kind}.txt").each do |line|
    n, l = line.split
    d = "#{kind}/label_%04d" % l.to_i
    FileUtils.mkdir_p d
    FileUtils.ln_sf n, "#{d}/#{File.basename(n)}"
  end
end

make('test')
make('train')
