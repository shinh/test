// gcc -g -shared dump_hook.c -Igcc -Iinclude -Iobj/gcc -Ilibcpp/include -o dump.so -fPIC
// /usr/local/stow/gcc-plugin/bin/gcc -ftree-plugin=./dump.so

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tree.h"
#include "tree-dump.h"
#include "tree-plugin.h"

void transform_ctrees(int argc, struct plugin_argument* argv, tree fndecl) {
  dump_node(fndecl, 0, stdout);
}
