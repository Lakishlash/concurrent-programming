#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    std::string msg = "Broadcast demo";
    int repeat = 2;

    for (int i = 1; i < argc; ++i)
    {
        std::string a(argv[i]);
        auto p = a.find('=');
        if (p == std::string::npos)
            continue;
        auto k = a.substr(0, p), v = a.substr(p + 1);
        if (k == "--msg")
            msg = v;
        if (k == "--repeat")
            repeat = std::stoi(v);
    }

    int len = 0;
    if (rank == 0)
        len = static_cast<int>(msg.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::string buf;
    buf.resize(len);
    if (rank == 0)
        std::copy(msg.begin(), msg.end(), buf.begin());
    if (len > 0)
        MPI_Bcast(buf.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int r = 0; r < repeat; ++r)
    {
        std::cout << "[bcast] rank " << rank << "/" << world << " received: " << buf << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << "Done.\n";
    MPI_Finalize();
    return 0;
}
