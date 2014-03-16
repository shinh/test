all: all_impl

include tls_test.mk
include tsx.mk
include deadlock.mk
include thumb_arm.mk
include arm_va.mk
include arm_syscall_detector.mk

all_impl: $(ALL)

clean:
	rm -rf $(CLEAN)
