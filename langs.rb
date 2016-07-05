#!/usr/bin/env ruby

require 'open-uri'

TBL_FILE = 'langs.tbl'

if !File.exist?(TBL_FILE)
  html = open('http://www.lingoes.net/en/translator/langcode.htm', &:read)
  m = {}
  html.scan(/<td>([a-z]{2})<\/td><td>(.*?)<\/td>/) do
    m[$1] = $2
  end

  File.open(TBL_FILE, 'w') do |ofile|
    m.sort.each do |k, v|
      ofile.puts "#{k} #{v.sub(/ .*/, '')}"
    end
  end
end

m = {}
File.readlines(TBL_FILE).each do |line|
  toks = line.split
  m[toks[0]] = toks[1]
  m[toks[1]] = toks[0]
  m[toks[1].upcase] = toks[0]
end

puts m[ARGV[0]]
