# pip3 install --user pyseccomp

try:
    import seccomp
except ImportError:
    import pyseccomp as seccomp

f = seccomp.SyscallFilter(defaction=seccomp.KILL)

f.add_rule(seccomp.ALLOW, "read")
f.add_rule(seccomp.ALLOW, "write")
f.add_rule(seccomp.ALLOW, "rt_sigreturn")
f.add_rule(seccomp.ALLOW, "rt_sigaction")
f.add_rule(seccomp.ALLOW, "sigaltstack")
f.add_rule(seccomp.ALLOW, "exit_group")

log = open("log", "w")

f.load()

log.write("OK")
log.flush()
print("log writen", flush=True)

malicious = open("malicious", "w")
