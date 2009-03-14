#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

const char* namedpipe_path = "/tmp/pipe";

void * conf_ipc_wthrd(void * arg)
{
	int pipefd = 0;
	char * confdata = "hoge\n";
	
	for ( ; ; ) {
		pipefd = open(namedpipe_path, O_WRONLY);
		if(pipefd < 0) {
			//errprint("error at opening named pipe\n");
			break;
		}

		if(write(pipefd, confdata, strlen(confdata)) != strlen(confdata)) {
			//errprint("error at writing confdata\n");
		}
		close(pipefd);
		//sleep(3);	/* caution */
	}

	return NULL;
}

int main() {
    mkfifo(namedpipe_path, 0755);
    conf_ipc_wthrd(NULL);
}
