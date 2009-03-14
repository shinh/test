int main() {
    int i;
    int* ip = &i;
    int const* icp = &i;
    int* const ipc = &i;
    const int* cip = &i;
    int& ir = i;
    int const& icr = i;
    //int& const irc = i;
    const int& cir = i;

    int j;
    ip = &j;
    icp = &j;
    //ipc = &j;
    cip = &j;
    ir = j;
    //icr = j;
    //cir = j;
}
