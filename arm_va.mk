ARM_VA_ALL=arm_va
ARM_VA_CFLAGS=-mthumb-interwork -g

ARM_VA_OBJS=arm_va_main.o arm_va_func.o
ARM_CC=arm-linux-gnueabihf-gcc-4.7

ALL+=$(ARM_VA_ALL)
CLEAN+=$(ARM_VA_ALL) $(ARM_VA_OBJS)

arm_va: $(ARM_VA_OBJS)
	$(ARM_CC) $(ARM_VA_CFLAGS) $^ -o $@

arm_va_func.o: arm_va_func.c
	$(ARM_CC) -c $(ARM_VA_CFLAGS) $< -o $@ -marm
arm_va_main.o: arm_va_main.c
	$(ARM_CC) -c $(ARM_VA_CFLAGS) $< -o $@ -marm
