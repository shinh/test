#!/usr/bin/env ruby
#
# Usage:
#
# $ ruby gen_fake_cuda.rb /usr/local/cuda/lib64 fake_cuda_dir
# $ ruby gen_fake_cuda.rb ~/.cudnn/active/cuda/lib64 fake_cuda_dir
# $ ruby gen_fake_cuda.rb /usr/lib/x86_64-linux-gnu/libcuda.so fake_cuda_dir
# $ LD_PRELOAD=$(echo fake_cuda_dir/lib*.so | sed 's/ /:/g') python3 sample_export.py
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

CUSTOM_FUNCS = {}

CUSTOM_FUNCS['cudaMalloc'] = %Q(
long cudaMalloc(void** p) {
  *p = malloc(1);
  RETURN 0;
})
CUSTOM_FUNCS['cudaHostAlloc'] = %Q(
long cudaHostAlloc(void** p, size_t s) {
  *p = malloc(s);
  RETURN 0;
})
CUSTOM_FUNCS['cudaMallocHost'] = %Q(
long cudaMallocHost(void** p, size_t s) {
  *p = malloc(s);
  RETURN 0;
})
CUSTOM_FUNCS['cudaFree'] = %Q(
long cudaFree(void* p) {
  free(p);
  RETURN 0;
})

CUSTOM_FUNCS['cudaGetDevice'] = %Q(
long cudaGetDevice(int* d) {
  *d = 0;
  RETURN 0;
})

CUSTOM_FUNCS['nvrtcGetPTXSize'] = %Q(
long nvrtcGetPTXSize(void* p, size_t* s) {
  *s = 1;
  RETURN 0;
})
CUSTOM_FUNCS['nvrtcGetPTX'] = %Q(
long nvrtcGetPTX(void* p, char* s) {
  *s = '\\0';
  RETURN 0;
})

CUSTOM_FUNCS['cuLinkComplete'] = %Q(
long cuLinkComplete(void* st, void** b, size_t* s) {
  *s = 0;
  RETURN 0;
})

if File.directory?(ARGV[0])
  sofiles = Dir.glob("#{ARGV[0]}/*.so").sort
else
  sofiles = [ARGV[0]]
end

sofiles.each do |so|
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
      c += %Q(fprintf(stderr, "#{soname}: #{name}\\n"); )
    end
    c += "return"

    case typ
    when 'A'
    when 'T'
      func_def = CUSTOM_FUNCS[name]
      if !func_def
        func_def = %Q(long #{name}() { RETURN 0; })
      end
      func_def.gsub!('RETURN', c)
      code << func_def
    when 'B'
      code << %Q(void* #{name};)
    else
      raise "Unsupported type: #{line}"
    end
  end

  File.open("#{dest}/#{soname}.c", 'w') do |of|
    code.each do |line|
      of.puts line
    end
  end

  system("gcc -shared -fPIC -g -O #{dest}/#{soname}.c -o #{dest}/#{soname}")
end
