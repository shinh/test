#include <elf.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

template <int B>
struct ELF {
  typedef Elf32_Ehdr Ehdr;
  typedef Elf32_Phdr Phdr;
  typedef Elf32_Dyn Dyn;
};

template<>
struct ELF<64> {
  typedef Elf64_Ehdr Ehdr;
  typedef Elf64_Phdr Phdr;
  typedef Elf64_Dyn Dyn;
};

template <int B>
void lld(int fd, const char* buf) {
  typedef typename ELF<B>::Ehdr Ehdr;
  typedef typename ELF<B>::Phdr Phdr;
  typedef typename ELF<B>::Dyn Dyn;
  const Ehdr* ehdr = reinterpret_cast<const Ehdr*>(buf);
  int phoff = static_cast<int>(ehdr->e_phoff);
  bool text_found = false;
  intptr_t text_addr = 0;
  intptr_t text_off = 0;
  for (int i = 0; i < ehdr->e_phnum; i++) {
    Phdr phdr;
    if (pread(fd, &phdr, sizeof(phdr), phoff) != sizeof(phdr)) {
      perror("pread (phdr)");
      exit(1);
    }

    phoff += sizeof(phdr);
    if (!text_found && phdr.p_type == PT_LOAD) {
      text_addr = phdr.p_vaddr;
      text_off = phdr.p_offset;
      text_found = true;
    }
    if (phdr.p_type != PT_DYNAMIC)
      continue;

    char* dynamic = static_cast<char*>(malloc(phdr.p_filesz));
    if (pread(fd, dynamic, phdr.p_filesz, phdr.p_offset) != phdr.p_filesz) {
      perror("pread (dynamic)");
      exit(1);
    }

    int off = static_cast<int>(phdr.p_offset);
    intptr_t straddr = 0;
    int strsz = 0;
    for (;;) {
      Dyn dyn;
      if (pread(fd, &dyn, sizeof(dyn), off) != sizeof(dyn)) {
        perror("pread (dyn)");
        exit(1);
      }
      off += sizeof(dyn);
      if (dyn.d_tag == DT_NULL)
        break;
      if (dyn.d_tag == DT_STRTAB) {
        straddr = dyn.d_un.d_val;
      } else if (dyn.d_tag == DT_STRSZ) {
        strsz = dyn.d_un.d_val;
      }
    }

    char* strtab = static_cast<char*>(malloc(strsz));
    if (pread(fd, strtab, strsz, straddr - text_addr + text_off) != strsz) {
      perror("pread (strtab)");
      exit(1);
    }

    off = static_cast<int>(phdr.p_offset);
    for (;;) {
      Dyn dyn;
      if (pread(fd, &dyn, sizeof(dyn), off) != sizeof(dyn)) {
        perror("pread (dyn)");
        exit(1);
      }
      off += sizeof(dyn);
      if (dyn.d_tag == DT_NULL)
        break;
      if (dyn.d_tag != DT_NEEDED)
        continue;

      puts(strtab + dyn.d_un.d_val /* addr? */);
    }
  }
}

int main(int argc, char* argv[]) {
  int fd = open(argv[1], O_RDONLY);
  if (fd < 0) {
    perror("open");
    return 1;
  }

  char buf[sizeof(Elf64_Ehdr)];
  if (read(fd, buf, sizeof(Elf64_Ehdr)) != sizeof(Elf64_Ehdr)) {
    perror("open");
    return 1;
  }

  if (strncmp(buf, ELFMAG, SELFMAG)) {
    fprintf(stderr, "not ELF\n");
    return 1;
  }

  if (buf[EI_CLASS] == ELFCLASS32) {
    lld<32>(fd, buf);
  } else if (buf[EI_CLASS] == ELFCLASS64) {
    lld<64>(fd, buf);
  } else {
    fprintf(stderr, "Unknown ELF class\n");
    return 1;
  }
  close(fd);
  return 0;
}
