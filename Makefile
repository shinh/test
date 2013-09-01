all: all_impl

include tls_test.mk
include tsx.mk
include deadlock.mk

all_impl: $(ALL)

clean:
	rm -rf $(CLEAN)
