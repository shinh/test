#!/usr/bin/env ruby

require 'pp'

if !ARGV[1]
  puts "Usage: #$0 <elf_binary> <symbol>"
  exit(1)
end

attr = nil
children = []
curparent = dwarf = []
curdep = 0
curnode = nil

typemap = {}

IO.popen("readelf -wi #{ARGV[0]}") do |pipe|
  while l = pipe.gets
    next if l !~ /^\s+<[(0-9a-f)]+>/
    if l =~ /^ <([ 0-9a-f]+)><[^>]*>: .*?\((.*?)\)/
      depth = $1.hex
      type = $2
      if curdep == depth
        n = curparent
        children = []
      else
        n = children
        children = []
        curdep = depth
      end
      attr = {}
      n << curnode = [type, attr, children]
    elsif l =~ /^  <([ 0-9a-f]+)>\s*(\S+)\s*:\s*(.*)/
      name = $2
      attr[name] = $3
      if name == 'DW_AT_name'
        typemap[$3.strip] = curnode
      end
    else
      raise l
    end
  end
end

pp dwarf
pp typemap[ARGV[1]]
