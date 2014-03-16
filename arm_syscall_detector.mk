ARM_SYSCALL_DETECTOR_ALL=arm_syscall_detector
ARM_SYSCALL_DETECTOR_CFLAGS=-g -W -Wall

ARM_SYSCALL_DETECTOR_OBJS=arm_syscall_detector.o

ALL+=$(ARM_SYSCALL_DETECTOR_ALL)
CLEAN+=$(ARM_SYSCALL_DETECTOR_ALL) $(ARM_SYSCALL_DETECTOR_OBJS)

arm_syscall_detector: $(ARM_SYSCALL_DETECTOR_OBJS)
	$(CXX) $(ARM_SYSCALL_DETECTOR_CFLAGS) $^ -o $@

arm_syscall_detector.o: arm_syscall_detector.cc
	$(CXX) -c $(ARM_SYSCALL_DETECTOR_CFLAGS) $< -o $@
