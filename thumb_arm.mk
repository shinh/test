THUMB_ARM_ALL=thumb_arm
THUMB_ARM_CFLAGS=-mthumb-interwork -g

THUMB_ARM_OBJS=arm_main.o arm_func.o thumb_func.o
ARM_CC=arm-linux-gnueabihf-gcc-4.7

ALL+=$(THUMB_ARM_ALL)
CLEAN+=$(THUMB_ARM_ALL) $(THUMB_ARM_OBJS)

thumb_arm: $(THUMB_ARM_OBJS)
	$(ARM_CC) $(THUMB_ARM_CFLAGS) $^ -o $@

thumb_func.o: thumb_func.c
	$(ARM_CC) -c $(THUMB_ARM_CFLAGS) $< -o $@ -mthumb
arm_func.o: arm_func.c
	$(ARM_CC) -c $(THUMB_ARM_CFLAGS) $< -o $@ -marm
arm_main.o: arm_main.c
	$(ARM_CC) -c $(THUMB_ARM_CFLAGS) $< -o $@ -marm
