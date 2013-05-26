ALL+=\
	tls_main tls_link_nopic tls_link_pic tls_link_pie tls_use_main \
	tls_use_local_exec.o tls_use_initial_exec.o \
	tls_use_local_dynamic.o tls_use_global_dynamic.o
CLEAN+=tls_*.o $(ALL)

BITS=32

# Base objects

tls_define_pic.o: tls_define.c
	$(CC) -m$(BITS) -fPIC -g -O -c $^ -o $@

tls_define_nopic.o: tls_define.c
	$(CC) -m$(BITS) -g -O -c $^ -o $@

tls_use.o: tls_use.c
	$(CC) -m$(BITS) -fPIC -g -O -c $^ -o $@

tls_use_main.o: tls_use_main.c
	$(CC) -m$(BITS) -fPIC -g -O -c $^ -o $@

tls_main.o : tls_main.c
	$(CC) -m$(BITS) -g -O -c tls_main.c -o $@

tls_main_pic.o : tls_main.c
	$(CC) -m$(BITS) -fPIC -g -O -c tls_main.c -o $@

# Compile time optimization

tls_main_incl.o: tls_main.c
	$(CC) -m$(BITS) -fPIC -g -O -c -include tls_define.c tls_main.c -o $@

tls_main: tls_main_incl.o
	$(CC) -m$(BITS) $^ -o $@

tls_link_nopic: tls_define_nopic.o tls_main.o
	$(CC) -m$(BITS) $^ -o $@

# Link time optimization

tls_link_pic: tls_define_pic.o tls_main.o
	$(CC) -m$(BITS) $^ -o $@

tls_link_pie: tls_define_pic.o tls_main_pic.o
	$(CC) -m$(BITS) -pie $^ -o $@

# Load time optimization?

tls_use.so: tls_use.o tls_define_pic.o
	$(CC) -m$(BITS) -shared $^ -o $@

tls_use_main: tls_use_main.o tls_use.so
	$(CC) -m$(BITS) -Wl,-rpath,. $^ -o $@

# TLS model attribute

tls_use_local_exec.o: tls_use.c
	$(CC) -m$(BITS) -fPIC -g -O -c $^ -o $@ -DTLS_MODEL='"local-exec"'

tls_use_initial_exec.o: tls_use.c
	$(CC) -m$(BITS) -fPIC -g -O -c $^ -o $@ -DTLS_MODEL='"initial-exec"'

tls_use_local_dynamic.o: tls_use.c
	$(CC) -m$(BITS) -fPIC -g -O -c $^ -o $@ -DTLS_MODEL='"local-dynamic"'

tls_use_global_dynamic.o: tls_use.c
	$(CC) -m$(BITS) -fPIC -g -O -c $^ -o $@ -DTLS_MODEL='"global-dynamic"'
