#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <libv4l2.h>
#include <linux/videodev2.h>

#include <SDL.h>

struct Buffer {
  char* start;
  size_t length;
  SDL_Surface* surface;
};

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <dev>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  int fd = v4l2_open(argv[1], O_RDWR);
  if (fd < 0) {
    perror("v4l2_open");
    exit(EXIT_FAILURE);
  }

  SDL_Init(SDL_INIT_VIDEO);

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
    exit(EXIT_FAILURE);
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
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (v4l2_ioctl(fd, VIDIOC_G_FMT, &format) < 0) {
    perror("v4l2_ioctl VIDIOC_G_FMT");
    exit(EXIT_FAILURE);
  }

  const struct v4l2_pix_format& pix_fmt = format.fmt.pix;
  char fmt_buf[5];
  fmt_buf[0] = (pix_fmt.pixelformat & 0xFF);
  fmt_buf[1] = (pix_fmt.pixelformat >> 8) & 0xFF;
  fmt_buf[2] = (pix_fmt.pixelformat >> 16) & 0xFF;
  fmt_buf[3] = (pix_fmt.pixelformat >> 24) & 0xFF;
  fmt_buf[4] = 0;
  printf("w=%u h=%u fmt=%s field=%u bypl=%u sz=%u cs=%u priv=%u\n",
         pix_fmt.width, pix_fmt.height, fmt_buf,
         pix_fmt.field, pix_fmt.bytesperline, pix_fmt.sizeimage,
         pix_fmt.colorspace, pix_fmt.priv);

  //format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
  if (v4l2_ioctl(fd, VIDIOC_S_FMT, &format) < 0) {
    perror("v4l2_ioctl VIDIOC_S_FMT");
    exit(EXIT_FAILURE);
  }
  fmt_buf[0] = (pix_fmt.pixelformat & 0xFF);
  fmt_buf[1] = (pix_fmt.pixelformat >> 8) & 0xFF;
  fmt_buf[2] = (pix_fmt.pixelformat >> 16) & 0xFF;
  fmt_buf[3] = (pix_fmt.pixelformat >> 24) & 0xFF;
  fmt_buf[4] = 0;
  const int W = pix_fmt.width;
  const int H = pix_fmt.height;
  printf("w=%u h=%u fmt=%s field=%u bypl=%u sz=%u cs=%u priv=%u\n",
         pix_fmt.width, pix_fmt.height, fmt_buf,
         pix_fmt.field, pix_fmt.bytesperline, pix_fmt.sizeimage,
         pix_fmt.colorspace, pix_fmt.priv);

  struct v4l2_input input;
  int input_index;
  if (v4l2_ioctl(fd, VIDIOC_G_INPUT, &input_index) < 0) {
    perror("v4l2_ioctl VIDIOC_G_INPUT");
    exit(EXIT_FAILURE);
  }

  memset(&input, 0, sizeof(input));
  input.index = input_index;

  if (v4l2_ioctl(fd, VIDIOC_ENUMINPUT, &input) < 0) {
    perror ("v4l2_ioctl VIDIOC_ENUMINPUT");
    exit(EXIT_FAILURE);
  }

  printf ("Current input: %s @%d\n", input.name, input_index);

  v4l2_std_id std_id;
  struct v4l2_standard standard;

  if (v4l2_ioctl(fd, VIDIOC_G_STD, &std_id) < 0) {
    /* Note when VIDIOC_ENUMSTD always returns EINVAL this
       is no video device or it falls under the USB exception,
       and VIDIOC_G_STD returning EINVAL is no error. */
    perror("v4l2_ioctl VIDIOC_G_STD");
    exit(EXIT_FAILURE);
  }

  memset (&standard, 0, sizeof(standard));
  standard.index = 0;

  while (v4l2_ioctl(fd, VIDIOC_ENUMSTD, &standard) == 0) {
    if (standard.id & std_id) {
      printf("Current video standard: %s\n", standard.name);
      errno = 0;
      break;
    }

    standard.index++;
  }

  /* EINVAL indicates the end of the enumeration, which cannot be
     empty unless this device falls under the USB exception. */
  if (errno == EINVAL /* || standard.index == 0 */) {
    perror("v4l2_ioctl VIDIOC_ENUMSTD");
    exit(EXIT_FAILURE);
  }

  v4l2_std_id new_std_id = V4L2_STD_NTSC_M_JP;
  if (v4l2_ioctl(fd, VIDIOC_S_STD, &new_std_id) < 0) {
    perror("v4l2_ioctl VIDIOC_S_STD");
    exit(EXIT_FAILURE);
  }

  struct v4l2_requestbuffers reqbuf;
  memset(&reqbuf, 0, sizeof (reqbuf));
  reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  reqbuf.memory = V4L2_MEMORY_MMAP;
  reqbuf.count = 20;

  if (v4l2_ioctl(fd, VIDIOC_REQBUFS, &reqbuf) < 0) {
    if (errno == EINVAL)
      fprintf(stderr,
              "Video capturing or mmap streaming is not supported\n");
    else
      perror("v4l2_ioctl VIDIOC_REQBUFS");
    exit(EXIT_FAILURE);
  }
  printf("Buffer count=%u type=%u memory=%u\n",
         reqbuf.count, reqbuf.type, reqbuf.memory);

  if (reqbuf.count < 5) {
    fprintf(stderr, "Not enough buffer memory: %u\n", reqbuf.count);
    exit(EXIT_FAILURE);
  }

  Buffer* buffers = (Buffer*)calloc(reqbuf.count, sizeof(*buffers));
  assert(buffers != NULL);

  printf("mmap:");
  for (size_t i = 0; i < reqbuf.count; i++) {
    struct v4l2_buffer buffer;
    memset(&buffer, 0, sizeof (buffer));
    buffer.type = reqbuf.type;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = i;
    if (v4l2_ioctl(fd, VIDIOC_QUERYBUF, &buffer) < 0) {
      perror("VIDIOC_QUERYBUF");
      exit(EXIT_FAILURE);
    }

    buffers[i].length = buffer.length; /* remember for munmap() */
    buffers[i].start = (char*)mmap(NULL, buffer.length,
                                   PROT_READ | PROT_WRITE, /* recommended */
                                   MAP_SHARED,             /* recommended */
                                   fd, buffer.m.offset);
    buffers[i].surface = SDL_CreateRGBSurfaceFrom(buffers[i].start,
                                                  W, H, 16, W*2,
                                                  31, 63 << 5, 31 << 11, 0);

    printf(" %lu:%p+%lu", i, buffers[i].start, buffers[i].length);
    if (MAP_FAILED == buffers[i].start) {
      /* If you do not exit here you should unmap() and free()
         the buffers mapped so far. */
      perror("mmap");
      exit(EXIT_FAILURE);
    }
  }
  puts("");

  for (size_t i = 0; i < reqbuf.count; i++) {
    struct v4l2_buffer buffer;
    memset(&buffer, 0, sizeof (buffer));
    buffer.type = reqbuf.type;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = i;

    if (v4l2_ioctl(fd, VIDIOC_QBUF, &buffer) < 0) {
      perror("VIDIOC_QBUF");
      exit(EXIT_FAILURE);
    }
  }

  if (v4l2_ioctl(fd, VIDIOC_STREAMON, &reqbuf.type) < 0) {
    perror("v4l2_ioctl VIDIOC_STREAMON");
    exit(EXIT_FAILURE);
  }

  SDL_Surface* scr = SDL_SetVideoMode(W, H, 16, SDL_SWSURFACE);
  //SDL_Surface* scr = SDL_SetVideoMode(W, H, 32, SDL_SWSURFACE);
  assert(scr);
  printf("Screen: pitch=%d bpp=%d rs=%d gs=%d bs=%d rm=%d gm=%d bm=%d\n",
         scr->pitch, scr->format->BitsPerPixel,
         scr->format->Rshift, scr->format->Gshift, scr->format->Bshift,
         scr->format->Rmask, scr->format->Gmask, scr->format->Bmask);

  //assert(scr->pitch == pix_fmt.bytesperline);

  bool done = false;
  while (!done) {
    struct v4l2_buffer buffer;
    memset(&buffer, 0, sizeof (buffer));
    buffer.type = reqbuf.type;
    buffer.memory = V4L2_MEMORY_MMAP;

    if (v4l2_ioctl(fd, VIDIOC_DQBUF, &buffer) < 0) {
      perror("VIDIOC_DQBUF");
      exit(EXIT_FAILURE);
    }

    //printf("%u\n", buffer.index);

    const Buffer& buf = buffers[buffer.index];
#if 0
#if 1
    for (size_t y = 0; y < pix_fmt.height; y++) {
      Uint16* src = (Uint16*)(buf.start + y * pix_fmt.bytesperline);
      Uint16* dst = (Uint16*)((char*)scr->pixels + y * scr->pitch);
      for (size_t x = 0; x < pix_fmt.width; x++) {
        Uint16 rgb = *src;
        Uint16 r = rgb & 31;
        Uint16 g = (rgb >> 5) & 63;
        Uint16 b = rgb >> 11;
#if 0
        *dst = SDL_MapRGB(scr->format, r << 3, g << 3, b << 3);
#else
        *dst = ((r << scr->format->Rshift) |
                (g << scr->format->Gshift) |
                (b << scr->format->Bshift));
#endif
        src++;
        dst++;
      }
    }
#else
    memcpy(scr->pixels, buf.start, pix_fmt.sizeimage);
#endif

#else
    SDL_BlitSurface(buf.surface, NULL, scr, NULL);
#endif

    if (v4l2_ioctl(fd, VIDIOC_QBUF, &buffer) < 0) {
      perror("VIDIOC_QBUF");
      exit(EXIT_FAILURE);
    }

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

#if 0
  //format.fmt.vbi.sample_format = V4L2_PIX_FMT_RGB555;
  format.fmt.vbi.sample_format = V4L2_PIX_FMT_BGR32;
  if (v4l2_ioctl(fd, VIDIOC_S_FMT, &format) < 0) {
    perror("v4l2_ioctl VIDIOC_S_FMT");
    exit(EXIT_FAILURE);
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
      exit(EXIT_FAILURE);
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
#endif

  for (size_t i = 0; i < reqbuf.count; i++)
    munmap(buffers[i].start, buffers[i].length);

  v4l2_close(fd);
}
