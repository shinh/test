THUMB_ARM_ALL=arm_va
THUMB_ARM_CFLAGS=-mthumb-interwork -g

THUMB_ARM_OBJS=arm_va_main.o arm_va_func.o
ARM_CC=arm-linux-gnueabihf-gcc-4.7

ALL+=$(THUMB_ARM_ALL)
CLEAN+=$(ALL) $(THUMB_ARM_OBJS)

arm_va: $(THUMB_ARM_OBJS)
	$(ARM_CC) $(THUMB_ARM_CFLAGS) $^ -o $@

arm_va_func.o: arm_va_func.c
	$(ARM_CC) -c $(THUMB_ARM_CFLAGS) $< -o $@ -marm
arm_va_main.o: arm_va_main.c
	$(ARM_CC) -c $(THUMB_ARM_CFLAGS) $< -o $@ -marm
