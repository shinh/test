#!/usr/bin/ruby
$:.push("#{ENV['HOME']}/bin")
require '__wraputil'
w3m_wrap_each("#{ENV['HOME']}/test/tanlog.rb", ['paths']) do |line|
  line =~ / in (.*?)@/
  path = CGI.escapeHTML($`)
  cmd = $1
  log = $'
  %Q(<a href="#{path}">#{path}</a> <a href="#{log}">#{cmd}</a>)
end
