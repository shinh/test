#!/usr/bin/env ruby
#
# inspired by https://github.com/omakoto/zenlog

require 'fileutils'
require 'tempfile'

TANLOG_DIR = '/tmp/tanlog'

ZSH_CODE = <<EOC
tanlog_begin() {
    export TANLOG_LOGFILE=$($HOME/test/tanlog.rb start $1)
}
tanlog_end() {
    $HOME/test/tanlog.rb end $TANLOG_LOGFILE
}
typeset -Uga preexec_functions
typeset -Uga precmd_functions
preexec_functions+=tanlog_begin
precmd_functions+=tanlog_end
EOC

Encoding.default_external = 'binary'
Encoding.default_internal = 'binary'

def rotate_log
  # TODO
end

def screen(args)
  if !system("screen #{args * ' '}")
    raise "command failed: #{args}"
  end
end

def raw_to_san(rawfile)
  rawfile.sub('/RAW/', '/')
end

def create_prev_links(logfile, dir)
  4.downto(1){|n|
    prev_link = "#{dir}/" + "P" * n
    if File.exist?(prev_link)
      File.rename(prev_link, prev_link + "P")
    end
  }
  FileUtils.ln_sf(logfile, "#{dir}/P")
end

def setup_cmd_link(logfile, cmd)
  arg0 = cmd.sub(/^[()\s]+/, '').split[0]
  arg0 = File.basename(arg0)
  ["#{TANLOG_DIR}/RAW/#{arg0}",
   "#{File.dirname(logfile)}/#{arg0}"].each do |cmddir|
    [[cmddir, logfile],
     [raw_to_san(cmddir), raw_to_san(logfile)]].each do |cd, lf|
      FileUtils.mkdir_p(cd)
      dest = "#{cd}/#{File.basename(lf)}"
      if File.exist?(dest)
        File.unlink(dest)
      end
      FileUtils.ln_s(lf, dest)

      create_prev_links(lf, cd)
    end
  end
end

def setup_log(cmd)
  now = Time.now
  date = now.strftime('%Y-%m-%d')
  logdir = "#{TANLOG_DIR}/RAW/#{date}"
  FileUtils.mkdir_p(logdir)
  FileUtils.mkdir_p(raw_to_san(logdir))

  FileUtils.rm_f("#{TANLOG_DIR}/.TODAY")
  FileUtils.ln_sf(raw_to_san(logdir), "#{TANLOG_DIR}/.TODAY")
  File.rename("#{TANLOG_DIR}/.TODAY", "#{TANLOG_DIR}/TODAY")

  time = now.strftime('%H:%M:%S')
  n = 0
  while true
    logfile = "#{logdir}/#{time}-#{n}.log"
    break if !File.exist? logfile
    n += 1
  end

  File.open(logfile, 'w') do |of|
    of.puts "$ #{cmd}"
  end

  screen(['-X', 'logfile', logfile])
  screen(['-X', 'log', 'on'])

  print logfile

  create_prev_links(raw_to_san(logfile), "#{TANLOG_DIR}/TODAY")
  setup_cmd_link(logfile, cmd)
end

def start_tanlog(args)
  setup_log(args[0])
end

def sanitize_log(rawfile)
  sanfile = raw_to_san(rawfile)

  File.open(rawfile) do |ifile|
    File.open(sanfile, 'w') do |of|
      ifile.each do |log|
        log.gsub!(/\a                        # Bell
                  | \e \x5B .*? [\x40-\x7E]  # CSI
                  | \e \x5D .*? \x07         # Set terminal title
                  | \e [\x40-\x5A\x5C\x5F]   # 2 byte sequence
                  /x, '')
        log.gsub!(/\s* \x0d* \x0a/x, "\x0a")  # Remove end-of-line CRs.
        log.gsub!(/ \s* \x0d /x, "\x0a")      # Replace orphan CRs with LFs.

        of.print log
      end
    end
  end
end

def end_tanlog(args)
  screen(['-X', 'log', 'off'])
  sanitize_log(args[0]) if args[0]
end

def show_recent_logs(args)
  logs = Dir.glob("#{TANLOG_DIR}/TODAY/P*").sort
  logs.reverse.each do |log|
    print File.read(log)
  end
end

cmd, *args = ARGV

case cmd
when 'start'
  exit if ENV['TERM'] !~ /screen/ || ENV['SSH_TTY']
  start_tanlog(args)
when 'end'
  exit if ENV['TERM'] !~ /screen/ || ENV['SSH_TTY']
  end_tanlog(args)
when 'recent'
  show_recent_logs(args)
else
  raise "Unknown tanlog command: #{cmd}"
end
