#include <stdio.h>
#include <unistd.h>

const int NUM_CPU = 2;

const char* RED = "\x1b[01;31m";
const char* GREEN = "\x1b[01;32m";
const char* RESET = "\x1b[0m";
const char SCREEN_RED[] = "\x05{+b r}";
const char SCREEN_GREEN[] = "\x05{+b g}";
const char SCREEN_RESET[] = "\x05{-}";

typedef long long int64;

typedef struct {
    int64 usertime, nicetime, systime, idletime, iowait, irq, softirq, steal;
    int64 total;
} CPU;

CPU CPU_read(FILE* fp) {
    CPU cpu;
    char buf[9];
    fscanf(fp, "%s%lld%lld%lld%lld%lld%lld%lld",
           buf,
           &cpu.usertime, &cpu.nicetime, &cpu.systime, &cpu.idletime,
           &cpu.iowait, &cpu.irq, &cpu.softirq, &cpu.steal);
    return cpu;
}

CPU CPU_diff(CPU* cur, CPU* prev) {
    CPU r;
    r.usertime = cur->usertime - prev->usertime;
    r.nicetime = cur->nicetime - prev->nicetime;
    r.systime = cur->systime - prev->systime;
    r.idletime = cur->idletime - prev->idletime;
    r.iowait = cur->iowait - prev->iowait;
    r.irq = cur->irq - prev->irq;
    r.softirq = cur->softirq - prev->softirq;
    r.steal = cur->steal - prev->steal;
    return r;
}

int64 CPU_total(CPU* cpu) {
    return (cpu->usertime + cpu->nicetime + cpu->systime + cpu->idletime +
            cpu->iowait + cpu->irq + cpu->softirq + cpu->steal);
}

typedef struct {
    int total, free, buffers, cached, swapcached, active, inactive,
        active_anon, inactive_anon, active_file, inactive_file,
        unevictable, mlocked, swap_total, swap_free, dirty,
        write_back, anon_pages, mapped, shmem,
        slab, sreclaimable, sunreclaim;
} Mem;

Mem Mem_read(FILE* fp) {
    Mem r;
    char buf[99];
    int i;
    for (i = 0; i < sizeof(r) / sizeof(int); i++) {
        fscanf(fp, "%s%d kB", buf, (int*)&r + i);
    }
    return r;
}

typedef struct {
    int read_completed, read_merged, read_sector, read_msec,
        write_completed, write_merged, write_sector, write_msec,
        io, io_msec, weighted_io_msec;
} Disk;

Disk Disk_read(FILE* fp) {
    Disk r;
    int major, minor;
    char buf[99];
    fscanf(fp, "%d%d%s%d%d%d%d%d%d%d%d%d%d%d",
           &major, &minor, buf,
           &r.read_completed, &r.read_merged, &r.read_sector, &r.read_msec,
           &r.write_completed, &r.write_merged, &r.write_sector, &r.write_msec,
           &r.io, &r.io_msec, &r.weighted_io_msec);
    return r;
}

typedef struct {
    int64 r_bytes, t_bytes;
} Net;

Net Net_read(FILE* fp) {
    Net r;
    int64 b;
    while (fgetc(fp) != ':') {}
    fscanf(fp, "%d%d%d%d%d%d%d%d",
           &r.r_bytes, &b, &b, &b, &b, &b, &b, &b);
    fscanf(fp, "%d%d%d%d%d%d%d%d",
           &r.t_bytes, &b, &b, &b, &b, &b, &b, &b);
    return r;
}

void print_percent(int value, int total, int green, int red) {
    int percent = value * 100 / total;
    if (percent >= red) {
        fputs(RED, stdout);
    } else if (percent >= green) {
        fputs(GREEN, stdout);
    }
    printf("%4d%%", percent);
    if (percent >= green) {
        fputs(RESET, stdout);
    }
}

void print_zero_percent() {
    print_percent(0, 1, 1, 1);
}

int main(int argc, char* argv[]) {
    int show_all = 1;
    int show_cpu = 0;
    int show_mem = 0;
    int show_disk = 0;
    int show_net = 0;
    int opt;
    while ((opt = getopt(argc, argv, "cmdns")) != -1) {
        switch (opt) {
        case 'c': show_all = 0; show_cpu = 1; break;
        case 'm': show_all = 0; show_mem = 1; break;
        case 'd': show_all = 0; show_disk = 1; break;
        case 'n': show_all = 0; show_net = 1; break;
        case 's':
            RED = SCREEN_RED;
            GREEN = SCREEN_GREEN;
            RESET = SCREEN_RESET;
        }
    }

    CPU prev_cpu;
    Disk prev_disk;
    Net prev_net;
    prev_cpu.usertime = -1;
    prev_disk.read_completed = -1;
    prev_net.r_bytes = -1;

    while (1) {
        {
            FILE* fp = fopen("/proc/stat", "rb");
            CPU cpu = CPU_read(fp);
            fclose(fp);

            fputs(" C", stdout);
            if (prev_cpu.usertime >= 0) {
                CPU diff = CPU_diff(&cpu, &prev_cpu);
                int64 total = CPU_total(&diff);
                if (total) {
                    int64 busy = total - diff.idletime;
                    print_percent(busy * NUM_CPU, total, 100, 180);
                    print_percent(diff.usertime * NUM_CPU, total, 100, 180);
                }
            } else {
                print_zero_percent();
                print_zero_percent();
            }
            prev_cpu = cpu;
        }

        {
            FILE* fp = fopen("/proc/meminfo", "rb");
            Mem mem = Mem_read(fp);
            fclose(fp);

            fputs(" M", stdout);
            print_percent(mem.total - mem.free -
                          mem.buffers - mem.cached - mem.sreclaimable,
                          mem.total, 50, 90);
            print_percent(mem.total - mem.free,
                          mem.total, 50, 90);
        }

        {
            FILE* fp = fopen("/proc/diskstats", "rb");
            Disk disk = Disk_read(fp);
            fclose(fp);

            fputs(" D", stdout);
            if (prev_disk.read_completed >= 0) {
                print_percent(disk.read_msec - prev_disk.read_msec, 1000,
                              50, 80);
                print_percent(disk.write_msec - prev_disk.write_msec, 1000,
                              50, 80);
            } else {
                print_zero_percent();
                print_zero_percent();
            }
            prev_disk = disk;
        }

        {
            FILE* fp = fopen("/proc/net/dev", "rb");
            char buf[256];
            fgets(buf, 255, fp);  // Inter-|
            fgets(buf, 255, fp);  //  face |
            fgets(buf, 255, fp);  //     lo:
            Net net = Net_read(fp);
            fclose(fp);

            fputs(" N", stdout);
            if (prev_net.r_bytes >= 0) {
                static const int MAX_BYTES_PER_SEC = (1 << 20) * 100 / 8;
                print_percent(net.r_bytes - prev_net.r_bytes,
                              MAX_BYTES_PER_SEC, 50, 80);
                print_percent(net.t_bytes - prev_net.t_bytes,
                              MAX_BYTES_PER_SEC, 50, 80);
            } else {
                print_zero_percent();
                print_zero_percent();
            }
            prev_net = net;
        }

        puts("");
        fflush(stdout);
        sleep(1);
    }
}
