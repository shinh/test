#!/usr/bin/env ruby

file = ARGV[0]
line = ARGV[1]

file = File.realpath(file)
git_dir = File.dirname(file)
until File.exist?(git_dir + "/.git")
  git_dir = File.dirname(git_dir)
  if git_dir.size <= 1
    raise "Not in git"
  end
end

Dir.chdir(git_dir) do
  url = `git remote get-url upstream 2> /dev/null`
  if url.empty?
    url = `git remote get-url origin 2> /dev/null`
  end
  if url.empty?
    raise "Not in git"
  end

  url.gsub!(/git@(.*)?:/, 'https://\1/')
  url.gsub!(/ssh:\/\/git@/, 'https://')
  url.gsub!(/\.git$/, '')
  url.chomp!

  branches = `git branch --list`
  branch = branches[/\s*(master|main|dev)$/, 1]

  rel = file[git_dir.size..-1]

  url = "#{url}/tree/#{branch}#{rel}"

  if line
    url += "#L#{line}"
  end

  puts url
end
