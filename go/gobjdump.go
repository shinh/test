package main

import "bufio"
import "debug/elf"
import "debug/gosym"
import "exec"
import "fmt"
import "os"
import "strconv"
import "strings"

func extractAddrFromLine(line string) uint64 {
	if len(line) < 8 {
		return 0;
	}
	i := strings.Index(line, ":");
	if i < 0 {
		return 0;
	}
	addr, e := strconv.Btoui64(strings.TrimSpace(line[0:i]), 16);
	if e != nil {
		return 0;
	}
	return addr;
}

func main() {
	// len(Args)
	f, e := elf.Open(os.Args[1]);
	if e != nil {
		println(e);
		return;
	}

	text := f.Section(".text");

	gopclntab := f.Section(".gopclntab");
	gopclndata, e := gopclntab.Data();
	if e != nil {
		println(e);
		return;
	}
	pclntab := gosym.NewLineTable(gopclndata, text.Addr);

	gosymtab := f.Section(".gosymtab");
	gosymdata, e := gosymtab.Data();
	symtab, e := gosym.NewTable(gosymdata, pclntab);
	if e != nil {
		println(e);
		return;
	}

	args := make([]string, 3);
	args[0] = "/usr/bin/objdump";
	args[1] = "-D";
	args[2] = os.Args[1];
	cmd, e := exec.Run(args[0], args, os.Environ(), exec.DevNull, exec.Pipe, exec.MergeWithStdout);
	if e != nil {
		println(e);
		return;
	}

	reader := bufio.NewReader(cmd.Stdout);
	for {
		line, e := reader.ReadString('\n');
		if e != nil {
			break;
		}

		addr := extractAddrFromLine(line);

		function := symtab.PCToFunc(addr);
		if function != nil && function.Entry == addr {
			fmt.Printf("%08x <%s.%s>:\n", addr, function.Sym.PackageName(), function.Sym.BaseName());
		}

		print(line);
	}

	f.Close();
}
