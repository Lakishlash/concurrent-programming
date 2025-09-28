#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // Flags: --msg=..., --tag=..., --repeat=...
    std::string msg = "SIT315 Activity01";
    int tag = 7, repeat = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto p = a.find('=');
        if (p == std::string::npos) continue;
        auto k = a.substr(0, p), v = a.substr(p + 1);
        if (k == "--msg")    msg    = v;
        if (k == "--tag")    tag    = std::stoi(v);
        if (k == "--repeat") repeat = std::stoi(v);
    }

    if (rank == 0) {
        int len = static_cast<int>(msg.size());
        for (int dest = 1; dest < world; ++dest) {
            for (int r = 0; r < repeat; ++r) {
                MPI_Send(&len, 1, MPI_INT,  dest, tag, MPI_COMM_WORLD);
                if (len) MPI_Send(msg.data(), len, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
            }
        }
        std::cout << "[pt2pt] rank 0/" << world
                  << " sent message to ranks 1.." << (world-1)
                  << " tag=" << tag << " repeat=" << repeat << "\n";
    } else {
        for (int r = 0; r < repeat; ++r) {
            int len = 0;
            MPI_Recv(&len, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string buf(len, '\0');
            if (len) MPI_Recv(buf.data(), len, MPI_CHAR, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "[pt2pt] rank " << rank << "/" << world << " received: " << buf << "\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Done.\n";
    MPI_Finalize();
    return 0;
}
