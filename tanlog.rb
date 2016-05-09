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

if ENV['TERM'] !~ /screen/ || ENV['SSH_TTY']
  exit
end

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

def setup_cmd_link(logfile, cmd)
  arg0 = cmd.split[0]
  cmddir = "#{TANLOG_DIR}/#{arg0}"
  FileUtils.mkdir_p(cmddir)
  FileUtils.ln_s(logfile, "#{cmddir}/#{File.basename(logfile)}")
end

def setup_log(cmd)
  now = Time.now
  date = now.strftime('%Y-%m-%d')
  logdir = "#{TANLOG_DIR}/#{date}"
  FileUtils.mkdir_p(logdir)

  FileUtils.rm_f("#{TANLOG_DIR}/.TODAY")
  FileUtils.ln_sf(logdir, "#{TANLOG_DIR}/.TODAY")
  File.rename("#{TANLOG_DIR}/.TODAY", "#{TANLOG_DIR}/TODAY")

  time = now.strftime('%H:%M:%S')
  n = 0
  while true
    logfile = "#{logdir}/#{time}-#{n}-raw.log"
    break if !File.exist? logfile
    n += 1
  end

  File.open(logfile, 'w') do |of|
    of.puts "$ #{cmd}"
  end

  screen(['-X', 'logfile', logfile])
  screen(['-X', 'log', 'on'])

  print logfile

  setup_cmd_link(logfile, cmd)
end

def start_tanlog(args)
  setup_log(args[0])
end

def sanitize_log(rawfile)
  sanfile = rawfile.sub('-raw.log', '-san.log')

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

cmd, *args = ARGV

case cmd
when 'start'
  start_tanlog(args)
when 'end'
  end_tanlog(args)
else
  raise "Unknown tanlog command: #{cmd}"
end
