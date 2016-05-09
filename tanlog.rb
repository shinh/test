#!/usr/bin/env ruby
#
# inspired by https://github.com/omakoto/zenlog
#
# _maybe_screen() {
#     if [ "x$TERM" = "xscreen" ]; then
#         if [ "x$SSH_TTY" = "x" ]; then
#             screen "$@"
#         fi
#     fi
# }
#
# tanlog_begin() {
#     local date=$(date '+%Y-%m-%d')
#     local logdir="/tmp/tanlog/${date}"
#     if [ ! -d $logdir ]; then
#         mkdir -p $logdir
#     fi
#     local time=$(date '+%H:%M:%S')
#     local n=0
#     local logfile=""
#     while [ -z $logfile -o -e $logfile ]; do
#         logfile="${logdir}/${time}-${n}-raw.log"
#         n=$(($n+1))
#     done
#     echo "\$ $1" > $logfile
#
#     export TANLOG_LOGFILE=$logfile
#     _maybe_screen -X logfile $logfile
#     _maybe_screen -X log on
# }
#
# tanlog_end() {
#     _maybe_screen -X log off
#     $HOME/test/tanlog.rb $TANLOG_LOGFILE &
# }
#
# preexec() {
#     tanlog_begin
# }
#
# precmd() {
#     tanlog_end
# }

def rotate_log
  # TODO
end

Encoding.default_external = 'binary'
Encoding.default_internal = 'binary'

if ARGV[0]
  rawfile = ARGV[0]
  sanfile = rawfile.sub('-raw.log', '-san.log')

  log = File.read(rawfile)
  log.gsub!(/\a                       # Bell
         | \e \x5B .*? [\x40-\x7E]  # CSI
         | \e \x5D .*? \x07         # Set terminal title
         | \e [\x40-\x5A\x5C\x5F]   # 2 byte sequence
/x, '')
  log.gsub!(/\s* \x0d* \x0a/x, "\x0a")  # Remove end-of-line CRs.
  log.gsub!(/ \s* \x0d /x, "\x0a")      # Replace orphan CRs with LFs.

  File.open(sanfile, 'w') do |of|
    of.print log
  end
end
