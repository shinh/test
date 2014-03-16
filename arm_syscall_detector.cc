#include <assert.h>
#include <elf.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <map>
#include <vector>

typedef unsigned char byte;

#ifndef ElfW
#define ElfW(type) Elf32_ ## type
#endif
#ifndef ELFW
#define ELFW(type) ELF32_ ## type
#endif

#define CHECK(b, ...)                           \
  do {                                          \
    if (!(b)) {                                 \
      fprintf(stderr, __VA_ARGS__);             \
      exit(1);                                  \
    }                                           \
  } while (0)

#if 0
static uint16_t ReadUint16(const void* ptr) {
  uint16_t v = *reinterpret_cast<uint16_t>(ptr);
#if defined(__LITTLE_ENDIAN__)
  v = (v >> 8) | ((v & 0xff) << 8);
#endif
  return v;
}

static uint32_t ReadUint32(const void* ptr) {
  uint32_t v = *reinterpret_cast<uint32_t>(ptr);
#if defined(__LITTLE_ENDIAN__)
  v = ((v >> 24) | ((v & 0xff0000) >> 8) |
       ((v & 0xff00) << 8) | ((v & 0xff) << 24));
#endif
  return v;
}
#endif

class ElfReader {
public:
  explicit ElfReader(const char* filename)
    : filename_(filename),
      dynamic_(NULL),
      dyn_sym_(NULL),
      dyn_sym_num_(0),
      dyn_str_(NULL),
      dyn_str_size_(0) {
    OpenAndMap();
    ParseElf();
  }

  std::vector<const ElfW(Phdr)*> texts() const { return texts_; }
  const ElfW(Phdr)* dynamic() const { return dynamic_; }
  const ElfW(Sym)* dyn_sym() const { return dyn_sym_; }
  size_t dyn_sym_num() const { return dyn_sym_num_; }
  const byte* dyn_str() const { return dyn_str_; }
  size_t dyn_str_size() const { return dyn_str_size_; }

  byte* GetByOffset(size_t offset) const {
    return mapped_ + offset;
  }

  byte* GetByAddr(size_t addr) const {
    for (size_t i = 0; i < loads_.size(); i++) {
      const ElfW(Phdr)* phdr = loads_[i];
      if (phdr->p_vaddr <= addr && addr < phdr->p_vaddr + phdr->p_filesz) {
        return GetByAddrInPhdr(phdr, addr);
      }
    }
    return NULL;
  }

  byte* GetByAddrInPhdr(const ElfW(Phdr)* phdr, size_t addr) const {
    return mapped_ + addr - phdr->p_vaddr + phdr->p_offset;
  }

private:
  void OpenAndMap() {
    fd_ = open(filename_, O_RDONLY);
    CHECK(fd_ >= 0, "open failed: %s\n", filename_);

    struct stat st;
    CHECK(fstat(fd_, &st) == 0, "fstat failed: %s\n", filename_);

    mapped_ = reinterpret_cast<byte*>(
        mmap(NULL, st.st_size,
             PROT_READ | PROT_WRITE, MAP_PRIVATE,
             fd_, 0));
    CHECK(mapped_ != MAP_FAILED, "mmap failed: %s\n", filename_);
  }

  void ParseElf() {
    ElfW(Ehdr)* ehdr = reinterpret_cast<ElfW(Ehdr*)>(mapped_);
    CHECK(!memcmp(ehdr->e_ident, ELFMAG, SELFMAG), "not ELF: %s\n", filename_);
    CHECK(ehdr->e_machine == EM_ARM, "not ARM: %s\n", filename_);

    ElfW(Phdr)* phdrs = reinterpret_cast<ElfW(Phdr)*>(mapped_ + ehdr->e_phoff);
    for (int i = 0; i < ehdr->e_phnum; i++) {
      const ElfW(Phdr)* phdr = &phdrs[i];
      if (phdr->p_type == PT_LOAD) {
        loads_.push_back(phdr);
        if (phdr->p_flags & PF_X) {
          texts_.push_back(phdr);
        }
      } else if (phdr->p_type == PT_DYNAMIC) {
        CHECK(!dynamic_, "multiple PT_DYNAMIC: %s\n", filename_);
        dynamic_ = phdr;

        for (ElfW(Dyn)* dyn = reinterpret_cast<ElfW(Dyn)*>(
                 GetByOffset(phdr->p_offset));
             dyn->d_tag != DT_NULL;
             dyn++) {
          const byte* ptr = GetByAddr(dyn->d_un.d_ptr);
          if (dyn->d_tag == DT_SYMTAB) {
            CHECK(ptr, "invalid DT_SYMTAB\n");
            dyn_sym_ = reinterpret_cast<const ElfW(Sym)*>(ptr);
          } else if (dyn->d_tag == DT_STRTAB) {
            CHECK(ptr, "invalid DT_STRTAB\n");
            dyn_str_ = GetByAddr(dyn->d_un.d_ptr);
          } else if (dyn->d_tag == DT_STRSZ) {
            dyn_str_size_ = dyn->d_un.d_val;
          } else if (dyn->d_tag == DT_HASH) {
            CHECK(ptr, "invalid DT_HASH\n");
            dyn_sym_num_ = reinterpret_cast<const uint32_t*>(ptr)[1];
          }
        }
      }
    }
    CHECK(dynamic_, "no PT_DYNAMIC: %s\n", filename_);
  }

  const char* filename_;
  int fd_;
  byte* mapped_;
  std::vector<const ElfW(Phdr)*> loads_;
  std::vector<const ElfW(Phdr)*> texts_;
  const ElfW(Phdr)* dynamic_;
  const ElfW(Sym)* dyn_sym_;
  size_t dyn_sym_num_;
  const byte* dyn_str_;
  size_t dyn_str_size_;
};

class SyscallDetector {
public:
  explicit SyscallDetector(const ElfReader* elf) : elf_(elf) {
  }

  ~SyscallDetector() {
    for (SymbolMap::const_iterator iter = symbols_.begin();
         iter != symbols_.end();
         ++iter) {
      delete iter->second;
    }
  }

  void Run() {
    ReadDynamic();

#if 0
    for (size_t i = 0; i < elf_->texts().size(); i++) {
      DetectSyscalls(elf_->texts()[i]);
    }
#endif
    DetectSyscalls();

    DumpRegions();
  }

private:
  enum SymbolType {
    SYMBOL_UNKNOWN,
    SYMBOL_DYNAMIC,
    SYMBOL_CALLED,
  };
  enum Mode {
    MODE_UNKNOWN,
    MODE_ARM,
    MODE_THUMB
  };

  struct Syscall {
    Syscall()
      : ptr(NULL),
        offset(0) {
    }

    byte* ptr;
    int offset;
    int imm;
  };

  struct Symbol {
    Symbol()
      : name(NULL),
        vaddr(0),
        size(0),
        type(SYMBOL_UNKNOWN),
        mode(MODE_UNKNOWN),
        parsed(false) {
    }

    const char* name;
    ElfW(Addr) vaddr;
    size_t size;
    SymbolType type;
    Mode mode;
    std::vector<Syscall> syscall_cands;
    bool parsed;
  };

  void ReadDynamic() {
    const ElfW(Sym)* sym_table = elf_->dyn_sym();
    for (size_t i = 0; i < elf_->dyn_sym_num(); i++) {
      const ElfW(Sym)* sym = &sym_table[i];
      if (ELFW(ST_TYPE)(sym->st_info) != STT_FUNC)
        continue;

      Symbol* symbol = new Symbol;
      symbol->name = reinterpret_cast<const char*>(
          elf_->dyn_str() + sym->st_name);
      symbol->vaddr = sym->st_value & ~1;
      symbol->size = sym->st_size;
      symbol->type = SYMBOL_DYNAMIC;
      symbol->mode = sym->st_value & 1 ? MODE_THUMB : MODE_ARM;
      std::pair<SymbolMap::const_iterator, bool> p = symbols_.insert(
          std::make_pair(elf_->GetByAddr(symbol->vaddr), symbol));
      if (!p.second) {
        CHECK(symbol->vaddr == p.first->second->vaddr &&
              symbol->mode == p.first->second->mode,
              "duplicated different symbol: %s vs %s\n",
              symbol->name, p.first->second->name);
        delete symbol;
      }
    }
  }

  void DetectSyscalls(/*const ElfW(Phdr)* phdr*/) {
    for (SymbolMap::const_iterator iter = symbols_.begin();
         iter != symbols_.end();
         ++iter) {
      byte* func = iter->first;
      Symbol* sym = iter->second;
      if (sym->mode == MODE_ARM)
        DetectSyscallsForArm(func, func + sym->size, sym);
      else if (sym->mode == MODE_THUMB)
        DetectSyscallsForThumb(func, func + sym->size, sym);
      else
        CHECK(false, "unknown execution mode: %d\n", sym->mode);
    }
  }

  void DetectSyscallsForArm(byte* begin, const byte* end, Symbol* sym) {
    for (byte* ptr = begin; ptr < end; ptr += 4) {
      if ((ptr[3] & 0xf) == 0xf) {
        Syscall syscall;
        syscall.ptr = ptr;
        syscall.offset = ptr - begin;
        syscall.imm = *reinterpret_cast<uint32_t*>(ptr) & 0xffffff;
        sym->syscall_cands.push_back(syscall);
      }
    }
  }

  void DetectSyscallsForThumb(byte* begin, const byte* end, Symbol* sym) {
    for (byte* ptr = begin; ptr < end; ptr += 2) {
      if (ptr[1] == 0xdf) {
        Syscall syscall;
        syscall.ptr = ptr;
        syscall.offset = ptr - begin;
        syscall.imm = ptr[0];
        sym->syscall_cands.push_back(syscall);
      }

      if ((ptr[1] >> 11) >= 29) {
        ptr += 2;
      }
    }
  }

  void DumpRegions() {
    for (SymbolMap::const_iterator iter = symbols_.begin();
         iter != symbols_.end();
         ++iter) {
      const Symbol* sym = iter->second;
      printf("%s: %08x-%08x %s (%s)\n",
             GetSymbolTypeStr(sym->type),
             sym->vaddr,
             static_cast<ElfW(Addr)>(sym->vaddr + sym->size),
             sym->name,
             GetModeStr(sym->mode));

      for (size_t i = 0; i < sym->syscall_cands.size(); i++) {
        const Syscall& syscall = sym->syscall_cands[i];
        printf("syscall: %08x (+%x) imm=%d\n",
               sym->vaddr + syscall.offset, syscall.offset, syscall.imm);
      }
    }
  }

  const char* GetSymbolTypeStr(SymbolType type) {
    if (type == SYMBOL_DYNAMIC)
      return "dynsym";
    if (type == SYMBOL_CALLED)
      return "called";
    CHECK(false, "unknown symbol type: %d\n", type);
  }

  const char* GetModeStr(Mode mode) {
    if (mode == MODE_ARM)
      return "ARM";
    if (mode == MODE_THUMB)
      return "Thumb";
    CHECK(false, "unknown execution mode: %d\n", mode);
  }

  const ElfReader* elf_;
  // From address accessible from this program to the information of
  // the symbol.
  typedef std::map<byte*, Symbol*> SymbolMap;
  SymbolMap symbols_;
};

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <elf>\n", argv[0]);
    return 1;
  }

  ElfReader elf(argv[1]);
  SyscallDetector detector(&elf);
  detector.Run();
}
