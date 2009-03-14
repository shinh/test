#include <elf.h>
int main() {
    printf("%d %d\n", sizeof(Elf32_Ehdr), sizeof(Elf64_Ehdr));
    printf("%d %d\n", sizeof(Elf32_Phdr), sizeof(Elf64_Phdr));
}
