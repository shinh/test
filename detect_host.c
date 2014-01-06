int main() {
  int fd, i;
  for (i = 0; i < 300; i++) {
    fd = open("detect_host.c", 0);
    if (fd < 0) {
      puts("mac");
      return 0;
    }
  }
  puts("linux");
  return 0;
}
