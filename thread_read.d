import std.stream;
import std.cstream;
import std.thread;

class ReaderThread : Thread {
    override int run() {
        char[] line = din.readLine();
        dout.writeLine(line);
        return 0;
    }
}

void main() {
    ReaderThread rt = new ReaderThread();
    rt.start();
    for (int i = 0; i < 100000; i++) {
        new Object;
    }
    rt.wait();
}
