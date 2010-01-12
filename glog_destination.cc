#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
using namespace std;
int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    // FLAGS_log_dir = "t";
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::INFO, "/var/tmp/another_destination.INFO");
    LOG(INFO) << "foobar";
    LOG(ERROR) << "foobar ERR";
}
