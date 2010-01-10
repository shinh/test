#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dwarf.h>
#include <elf.h>

#include <string>
#include <vector>

using namespace std;

#ifdef __x86_64__
# define ElfW(x) Elf64##_##x
#else
// There are only x86-64 and x86 in the world!
# define ElfW(x) Elf32##_##x
#endif

typedef unsigned long addr_t;
typedef unsigned char ubyte;
typedef unsigned int uint;
typedef unsigned long ulong;

void error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    exit(1);
}

char* read_file(const char* file) {
    FILE* fp = fopen(file, "rb");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buf = (char*)malloc(size);
    fread(buf, 1, size, fp);
    return buf;
}

ulong uleb128(char*& p) {
    ulong r = 0;
    int s = 0;
    for (;;) {
        ubyte b = *(ubyte*)p++;
        if (b < 0x80) {
            r += b << s;
            break;
        }
        r += (b & 0x7f) << s;
        s += 7;
    }
    return r;
}

long sleb128(char*& p) {
    long r = 0;
    int s = 0;
    for (;;) {
        ubyte b = *(ubyte*)p++;
        if (b < 0x80) {
            if (b & 0x40) {
                r -= (0x80 - b) << s;
            }
            else {
                r += (b & 0x3f) << s;
            }
            break;
        }
        r += (b & 0x7f) << s;
        s += 7;
    }
    return r;
}

string parse_filename(char*& p,
                      const vector<const char*>& include_directories) {
    const char* filename = p;
    while (*p) p++;
    p++;
    ulong dir = uleb128(p);
    const char* dirname = include_directories[dir];
    // last modified.
    uleb128(p);
    // size of the file.
    uleb128(p);
    if (dirname[0]) {
        return string(dirname) + "/" + filename;
    } else {
        return filename;
    }
}

void dump_debug_line_cu(char*& debug_line, unsigned long size) {
    char* p = debug_line;

    ulong unit_length = *(uint*)p;
    p += sizeof(uint);
    if (unit_length == 0xffffffff) {
        unit_length = *(ulong*)p;
        p += sizeof(ulong);
    }

    char* cu_end = p + unit_length;

    int dwarf_version = *(unsigned short*)p;
    p += 2;

    uint header_length = *(uint*)p;
    p += sizeof(uint);

    char* cu_start = p + header_length;

    uint minimum_instruction_length = *(ubyte*)p;
    p++;

    bool default_is_stmt = *(ubyte*)p;
    p++;

    int line_base = *(char*)p;
    p++;

    uint line_range = *(ubyte*)p;
    p++;

    uint opcode_base = *(ubyte*)p;
    p++;

    vector<int> standard_opcode_lengths(opcode_base);
    for (int i = 1; i < opcode_base; i++) {
        standard_opcode_lengths[i] = *(ubyte*)p;
        p++;
    }

    vector<const char*> include_directories;
    include_directories.push_back("");
    while (*p) {
        include_directories.push_back(p);
        //printf("%s\n", p);
        while (*p) p++;
        p++;
    }
    p++;

    vector<string> filenames;
    filenames.push_back("");
    while (*p) {
        filenames.push_back(parse_filename(p, include_directories));
        //printf("%s\n", filenames.back().c_str());
    }
    p++;

    // The registers.
    addr_t addr = 0;
    uint file = 1;
    uint line = 1;
    uint column = 0;
    bool is_stmt = default_is_stmt;
    bool basic_block = false;
    bool end_sequence = false;
    bool prologue_end = false;
    bool epilogue_begin = false;
    uint isa = 0;

#define DUMP_LINE()                                                 \
    do {                                                            \
        printf("%s:%d: %p\n", filenames[file].c_str(), line, addr); \
        basic_block = prologue_end = epilogue_begin = false;        \
    } while (0)

    //printf("%d %d %d\n", p-debug_line, unit_length, header_length);
    if (p != cu_start) {
        error("Unexpected header size\n");
    }

    while (p < cu_end) {
        ulong a;
        ubyte op = *p++;
        switch (op) {
        case DW_LNS_copy:
            DUMP_LINE();
            break;
        case DW_LNS_advance_pc:
            a = uleb128(p);
            addr += a;
            break;
        case DW_LNS_advance_line: {
            long a = sleb128(p);
            line += a;
            //printf("DW_LNS_advance_line %ld => %d\n", a, line);
            break;
        }
        case DW_LNS_set_file:
            file = uleb128(p);
            break;
        case DW_LNS_set_column:
            column = uleb128(p);
            break;
        case DW_LNS_negate_stmt:
            is_stmt = !is_stmt;
            break;
        case DW_LNS_set_basic_block:
            basic_block = true;
            break;
        case DW_LNS_const_add_pc:
            a = ((255 - opcode_base) / line_range) * minimum_instruction_length;
            addr += a;
            break;
        case DW_LNS_fixed_advance_pc:
            a = *(ubyte*)p++;
            addr += a;
            break;
        case DW_LNS_set_prologue_end:
            prologue_end = true;
            break;
        case DW_LNS_set_epilogue_begin:
            epilogue_begin = true;
            break;
        case DW_LNS_set_isa:
            isa = uleb128(p);
            break;
        case 0:
            a = *(ubyte*)p++;
            op = *p++;
            //printf("extended op: %d size=%d\n", op, a);
            switch (op) {
            case DW_LNE_end_sequence:
                end_sequence = true;
                DUMP_LINE();
                addr = 0;
                file = 1;
                line = 1;
                column = 0;
                is_stmt = default_is_stmt;
                end_sequence = false;
                isa = 0;
                break;
            case DW_LNE_set_address:
                addr = *(ulong*)p;
                p += sizeof(ulong);
                break;
            case DW_LNE_define_file:
                filenames.push_back(parse_filename(p, include_directories));
                break;
            default:
                error("Unknown extended opcode: %d\n", op);
            }
            break;
        default: {
            a = op - opcode_base;
            uint addr_incr = (a / line_range) * minimum_instruction_length;
            int line_incr = line_base + (a % line_range);
            addr += addr_incr;
            line += line_incr;
            //printf("special: addr +%d => %p, line +%d => %d\n",
            //       addr_incr, addr, line_incr, line);
            DUMP_LINE();
        }
        }
    }
    debug_line = p;
}

void dump_debug_line(char* debug_line, unsigned long size) {
    char* end = debug_line + size;
    while (debug_line < end) {
        dump_debug_line_cu(debug_line, size);
    }
    if (debug_line != end) {
        error("Unexpected size of debug_line\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        error("Usage: %s ELF-binary\n", argv[0]);
    }

    char* file = read_file(argv[1]);

    ElfW(Ehdr)* ehdr = (ElfW(Ehdr)*)file;
    ElfW(Shdr)* shdr = (ElfW(Shdr)*)(file + ehdr->e_shoff);

    ElfW(Shdr)* shstr_shdr = shdr + ehdr->e_shstrndx;
    char* shstr = file + shstr_shdr->sh_offset;

    ElfW(Shdr)* debug_line_shdr = 0;
    for (int i = 0; i < ehdr->e_shnum; i++) {
        char* name = shstr + shdr[i].sh_name;
        if (!strcmp(name, ".debug_line")) {
            debug_line_shdr = shdr + i;
            break;
        }
    }

    if (!debug_line_shdr) {
        error("no debug info?\n");
    }

    char* debug_line = file + debug_line_shdr->sh_offset;

    dump_debug_line(debug_line, debug_line_shdr->sh_size);

    free(file);
}
