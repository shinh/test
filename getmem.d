import std.moduleinit;

void main() {
    foreach (ModuleInfo mi; ModuleInfo.modules) {
        foreach (ClassInfo ci; mi.localClasses) {
            printf("%.*s\n", ci.name);
            foreach(const MemberInfo m; ci.getMembers(null)) {
                printf("%p\n", m);
            }
        }
    }
}
