#include <errno.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/inotify.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <map>
#include <queue>
#include <string>

#include <glog/logging.h>

using namespace google;
using namespace std;

void do_nothing(int unused) {
}

int add_watch_dir(int fd, const char* dir, map<int, string>* watches) {
    int wd = inotify_add_watch(fd, dir, IN_CREATE);
    PCHECK(wd >= 0) << "inotify_add_watch for " << dir;
    CHECK(watches->insert(make_pair(wd,dir)).second);
    return wd;
}

void read_events(int fd, string* buf) {
    char tmp[16384];
    ssize_t len = read(fd, tmp, sizeof(tmp));
    if (len < 0) {
        PCHECK(errno == EINTR) << "read";
        return;
    }
    if (len == 0) {
        return;
    }

    buf->insert(buf->end(), tmp, tmp + len);
}

void process_events(const map<int, string>& watches, string* buf) {
    const struct inotify_event* ev;
    size_t len_offset = offsetof(struct inotify_event, len) + sizeof(ev->len);
    while (buf->size() >= len_offset) {
        ev = reinterpret_cast<const struct inotify_event*>(buf->data());
        size_t event_size =  offsetof(struct inotify_event, name) + ev->len;
        if (buf->size() < event_size) {
            break;
        }

        map<int, string>::const_iterator found = watches.find(ev->wd);
        CHECK(found != watches.end());
        printf("%s/%s\n", found->second.c_str(), ev->name);

        buf->erase(0, event_size);
    }
}

void watch_child() {
    int fd = inotify_init ();
    PCHECK(fd >= 0) << "inotify_init";

    map<int, string> watches;
    add_watch_dir(fd, ".", &watches);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = &do_nothing;
    sa.sa_flags = SA_NOCLDSTOP;
    PCHECK(sigaction(SIGCHLD, &sa, NULL) >= 0) << "sigaction";

    string buf;
    while (true) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        int r = select(fd + 1, &fds, NULL, NULL, NULL);
        if (r < 0) {
            PCHECK(errno == EINTR) << "select";
            break;
        }
        if (FD_ISSET(fd, &fds)) {
            read_events(fd, &buf);
            process_events(watches, &buf);
        }
    }

    close(fd);
    PCHECK(wait(NULL) >= 0) << "wait";
}

int main(int argc, char* argv[]) {
    if (fork()) {
        google::InitGoogleLogging(argv[0]);
        watch_child();
    } else {
        argc--;
        argv++;
        execv(argv[0], argv);
    }
    return 0;
}
