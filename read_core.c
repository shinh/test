#include <assert.h>
#include <elf.h>
#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

char* read_file(const char* file) {
    FILE* fp = fopen(file, "rb");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buf = (char*)malloc(size);
    fread(buf, 1, size, fp);
    return buf;
}

int main(int argc, const char* argv[]) {
    char* core = read_file(argv[1]);
    Elf64_Ehdr* ehdr = (Elf64_Ehdr*)core;
    Elf64_Phdr* phdr = (Elf64_Phdr*)(core + ehdr->e_phoff);
    Elf64_Nhdr* note = 0;
    for (int i = 0; i < ehdr->e_phnum; ++i) {
        if (phdr[i].p_type == PT_NOTE) {
            note = (Elf64_Nhdr*)(core + phdr[i].p_offset);
            break;
        }
    }
    assert(note);

    while (1) {
        if (note->n_type == NT_SIGINFO) {
            break;
        }
        uintptr_t nxt = (uintptr_t)note + sizeof(Elf64_Nhdr) + note->n_namesz + note->n_descsz;
        note = (Elf64_Nhdr*)((nxt + 3) & ~3);
    }

    uintptr_t nxt = (uintptr_t)note + sizeof(Elf64_Nhdr) + note->n_namesz;
    siginfo_t* siginfo = (siginfo_t*)((nxt + 3) & ~3);
    printf("%p %p\n", siginfo->si_ptr, siginfo->si_addr);
}
