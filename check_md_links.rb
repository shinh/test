#!/usr/bin/env ruby

`git ls-files | grep '.md$'`.split.each do |mdfile|
  md = File.read(mdfile)
  md.scan(/\]\((.*)\)/) do
    target = $1
    next if target =~ /^https?:\/\//
    next if target =~ /^#/

    path = "#{File.dirname(mdfile)}/#{target}"
    if !File.exist?(path)
      puts "#{target} from #{mdfile}"
    end
  end
end
