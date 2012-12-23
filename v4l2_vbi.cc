#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <libv4l2.h>
#include <linux/videodev2.h>

#include <SDL.h>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <dev>\n", argv[0]);
    exit(1);
  }

  int fd = v4l2_open(argv[1], O_RDONLY);
  if (fd < 0) {
    perror("v4l2_open");
    exit(1);
  }

/*
  $2 = {
    driver = "em28xx\000\000\000\000\000\000\000\000\000",
    card = "EM2860/TVP5150 Reference Design",
    bus_info = "usb-0000:00:1d.0-1.1", '\000' <repeats 11 times>,
    version = 198154,
    capabilities = 84017233,
    device_caps = 0,
    reserved = {0, 0, 0}
  }
*/

  struct v4l2_capability cap;
  if (v4l2_ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
    perror("v4l2_ioctl VIDIOC_QUERYCAP");
    exit(1);
  }

/*
  vbi = {
    sampling_rate = 13500000,
    offset = 0,
    samples_per_line = 720,
    sample_format = 1497715271,
    start = {6, 318},
    count = {18, 18},
    flags = 0,
    reserved = {0, 0}
  },
*/

  struct v4l2_format format;
  format.type = V4L2_BUF_TYPE_VBI_CAPTURE;
  if (v4l2_ioctl(fd, VIDIOC_G_FMT, &format) < 0) {
    perror("v4l2_ioctl VIDIOC_G_FMT");
    exit(1);
  }

  //format.fmt.vbi.sample_format = V4L2_PIX_FMT_RGB555;
  format.fmt.vbi.sample_format = V4L2_PIX_FMT_BGR32;
  if (v4l2_ioctl(fd, VIDIOC_S_FMT, &format) < 0) {
    perror("v4l2_ioctl VIDIOC_S_FMT");
    exit(1);
  }

  struct v4l2_vbi_format& vbi_fmt = format.fmt.vbi;
  printf("rate=%d spl=%d cnt=%d,%d\n",
         vbi_fmt.sampling_rate, vbi_fmt.samples_per_line,
         vbi_fmt.count[0], vbi_fmt.count[1]);

  const int W = vbi_fmt.samples_per_line;
  const int H = 480;

  SDL_Init(SDL_INIT_VIDEO);

  //SDL_Surface* scr = SDL_SetVideoMode(W, H, 15, SDL_SWSURFACE);
  SDL_Surface* scr = SDL_SetVideoMode(W, H, 32, SDL_SWSURFACE);
  assert(scr);

  //const int bytes_per_pixel = 2;
  const int bytes_per_pixel = 4;

  ssize_t buf_size =
    (vbi_fmt.count[0] + vbi_fmt.count[1]) * W * bytes_per_pixel;
  char* buf = (char*)malloc(buf_size);

  int y[2];
  y[0] = vbi_fmt.start[0];
  y[1] = vbi_fmt.start[1];

  int t = 0;
  bool done = false;

  while (!done) {
    printf("#%d y0=%d y1=%d\n", t++, y[0], y[1]);

    if (v4l2_read(fd, buf, buf_size) != buf_size) {
      perror("v4l2_read");
      exit(1);
    }

    char* src = buf;
    for (int i = 0; i < 2; i++) {
      for (size_t j = 0; j < vbi_fmt.count[i]; j++) {
        char* dst = (char*)scr->pixels + scr->pitch * y[i];
        size_t sz = W * bytes_per_pixel;
        memcpy(dst, src, sz);
        src += sz;
        y[i] = (y[i] + 1) % H;
      }
    }

    assert(src == buf + buf_size);

    if (t % 26 == 0) {
      SDL_Event ev;
      while (SDL_PollEvent(&ev)) {
        switch (ev.type) {
        case SDL_QUIT:
          done = true;
          break;
        }
      }
      SDL_Flip(scr);
    }
  }

  v4l2_close(fd);
}
