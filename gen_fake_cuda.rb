#!/usr/bin/env ruby
#
# Usage:
#
# $ ruby gen_fake_cuda.rb /usr/local/cuda/lib64 fake_cuda_dir
#

require 'fileutils'

if ARGV[0] == '--trace'
  trace = true
  ARGV.shift
end

dest = ARGV[1]
if !dest
  STDERR.puts "Usage: #{$0} /usr/local/cuda/lib64 fake_cuda_dir"
  exit 1
end

FileUtils.mkdir_p dest

Dir.glob("#{ARGV[0]}/*.so").sort.each do |so|
  puts so
  soname = File.basename(so)

  code = []
  code << %Q(#include <stdio.h>)
  code << %Q(#include <stdlib.h>)
  IO.popen("nm -D #{so}").each do |line|
    toks = line.split
    next if toks.size != 3

    typ = toks[1]
    name = toks[2]

    c = ""
    if trace
      c += %Q(fprintf(stderr, "#{soname}: #{name}\\n");)
    end
    c += "return 0;"

    case typ
    when 'A'
    when 'T'
      if name =~ /^cudaMalloc$/
        code << %Q(long #{name}(void** p) { *p = malloc(1); #{c}; })
      elsif name =~ /^cudaHostAlloc$/
        code << %Q(long #{name}(void** p, size_t s) { *p = malloc(s); #{c}; })
      elsif name =~ /^cudaMallocHost$/
        code << %Q(long #{name}(void** p, size_t s) { *p = malloc(s); #{c}; })
      elsif name =~ /^cudaFree$/
        code << %Q(long #{name}(void* p) { free(p); #{c}; })
      else
        code << %Q(long #{name}() { #{c}; })
      end
    else
      raise "Unsupported type: #{typ}"
    end
  end

  File.open("#{dest}/#{soname}.c", 'w') do |of|
    code.each do |line|
      of.puts line
    end
  end

  system("gcc -shared -fPIC -g -O #{dest}/#{soname}.c -o #{dest}/#{soname}")
end
