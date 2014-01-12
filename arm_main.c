void arm_func(void);
void thumb_func(void);
int main() {
  thumb_func();
  arm_func();

  asm("blx %0":: "r"(arm_func));
  asm("blx %0":: "r"(thumb_func));

  return 0;
}
