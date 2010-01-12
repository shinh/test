#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
using namespace std;
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, true);
  cout << "hogehoge" << endl;
}

