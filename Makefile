all: all_impl

include tls_test.mk
include tsx.mk

all_impl: $(ALL)

clean:
	rm -rf $(CLEAN)
