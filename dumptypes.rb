#!/usr/bin/env ruby

require 'pp'

if !ARGV[1]
  puts "Usage: #$0 <elf_binary> <symbol>"
  exit(1)
end

nodes = []
curdep = 0

valmap = {}
typemap = {}

IO.popen("readelf -wi #{ARGV[0]}") do |pipe|
  while l = pipe.gets
    next if l !~ /^\s+<[(0-9a-f)]+>/
    l.strip!
    if l =~ /^\s*<([ 0-9a-f]+)><([^>]*)>: .*?\((.*?)\)/
      depth = $1.hex
      index = $2.hex
      type = $3

      nnode = [type, {}, []]
      if depth > 0
        curnode = nodes[depth-1]
        curnode[2] << nnode
      end

      valmap[index] = nnode
      nodes[depth] = nnode
      curdep = depth
    elsif l =~ /^\s*<([ 0-9a-f]+)>\s+(\S+)\s*:\s*(.*)/
      name = $2
      value = $3.sub(/^.*:/, '').strip

      if value =~ /^<0x([ 0-9a-f]+)>$/
        value = valmap[$1.hex]
      end

      curnode = nodes[curdep]
      attr = curnode[1]
      attr[name] = value

      type = curnode[0]
      if name == 'DW_AT_name' && (type == 'DW_TAG_typedef' ||
                                  type =~ /^DW_TAG_.*_type$/)
        sym = value
        typemap[sym] = curnode
      end
    else
      raise l
    end
  end
end

def get_name(n)
  n[1]['DW_AT_name']
end

def get_qualified_name(n)
  get_name(n)
end

def dump_cnode(n)
  children = n[2]
  children.each do |c|
    dump_cnode(c)
  end

  case n[0]
  when 'DW_TAG_typedef'
    orig_name = get_qualified_name(n[1]['DW_AT_type'])
    alias_name = get_name(n)
    puts "typedef #{orig_name} #{alias_name};"
  when 'DW_TAG_structure_type'
    
  end
end

#pp nodes[0]
if typemap[ARGV[1]]
  dump_cnode(typemap[ARGV[1]])
end
pp typemap[ARGV[1]]
