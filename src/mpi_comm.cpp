// src/mpi_comm.cpp
// SIT315 Module 3 – Activity 01: MPI communication patterns
// Modes:
//   --mode=pt2pt : master sends to each worker via MPI_Send/MPI_Recv
//   --mode=bcast : master broadcasts to all via MPI_Bcast
//
// Flags (no hardcoding):
//   --msg="..."      : message text (default: "Hello from SIT315 – Activity 01")
//   --tag=INT        : MPI tag for point-to-point mode (default: 0)
//   --repeat=INT     : repeat count for printing/sending (default: 1)
//   --help           : print usage
//
// Build (will run in Step 2):
//   mpic++ -O2 -std=c++17 src/mpi_comm.cpp -o build/mpi_comm
//
// Run examples (Step 3 later, do NOT run now):
//   mpirun -np 4 ./build/mpi_comm --mode=pt2pt --msg="SIT315!" --tag=7 --repeat=2
//   mpirun -np 4 ./build/mpi_comm --mode=bcast  --msg="Broadcast demo" --repeat=1

#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>

namespace
{
    struct Args
    {
        std::string mode = "pt2pt"; // "pt2pt" or "bcast"
        std::string msg = "Hello from SIT315 – Activity 01";
        int tag = 0;
        int repeat = 1;
        bool help = false;
    };

    static inline bool starts_with(const std::string &s, const char *pref)
    {
        return s.rfind(pref, 0) == 0;
    }

    Args parse_args(int argc, char **argv)
    {
        Args a;
        for (int i = 1; i < argc; ++i)
        {
            std::string tok(argv[i]);
            if (tok == "--help" || tok == "-h")
            {
                a.help = true;
                continue;
            }

            auto eq = tok.find('=');
            if (eq == std::string::npos)
                continue;
            std::string key = tok.substr(0, eq);
            std::string val = tok.substr(eq + 1);

            if (key == "--mode")
                a.mode = val;
            if (key == "--msg")
                a.msg = val;
            if (key == "--tag")
                a.tag = std::stoi(val);
            if (key == "--repeat")
                a.repeat = std::stoi(val);
        }
        return a;
    }

    void print_usage(int rank)
    {
        if (rank != 0)
            return;
        std::cout <<
            R"(Usage: mpirun -np <N> ./mpi_comm --mode=<pt2pt|bcast> [--msg="..."] [--tag=INT] [--repeat=INT]

Examples:
  mpirun -np 4 ./mpi_comm --mode=pt2pt --msg="SIT315!" --tag=7 --repeat=2
  mpirun -np 4 ./mpi_comm --mode=bcast  --msg="Broadcast demo" --repeat=1
)";
    }

    // Broadcast a string from root to all ranks (safe, no fixed buffers).
    std::string bcast_string(const std::string &src, int root, MPI_Comm comm)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);

        int len = static_cast<int>(src.size());
        // 1) broadcast length
        MPI_Bcast(&len, 1, MPI_INT, root, comm);

        // 2) broadcast data (no null terminator necessary)
        std::vector<char> buf(len);
        if (rank == root && len > 0)
        {
            std::memcpy(buf.data(), src.data(), len);
        }
        if (len > 0)
        {
            MPI_Bcast(buf.data(), len, MPI_CHAR, root, comm);
        }

        return std::string(buf.begin(), buf.end());
    }

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = -1, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // Parse only once (all ranks have the same argv) but then broadcast simple scalars
    Args args = parse_args(argc, argv);
    if (args.help)
    {
        print_usage(rank);
        MPI_Finalize();
        return 0;
    }

    // Validate mode on root; convert to integer code for broadcast consistency
    int mode_code = 0; // 0=pt2pt, 1=bcast
    if (rank == 0)
    {
        if (args.mode == "pt2pt")
            mode_code = 0;
        else if (args.mode == "bcast")
            mode_code = 1;
        else
        {
            std::cerr << "[rank 0] Error: --mode must be 'pt2pt' or 'bcast'\n";
            print_usage(rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (args.repeat < 1)
            args.repeat = 1;
    }

    // Share mode, tag, repeat with everyone (no hardcoding on other ranks)
    int shared_tag = args.tag;
    int shared_repeat = args.repeat;
    MPI_Bcast(&mode_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&shared_tag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&shared_repeat, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Mode dispatch
    if (mode_code == 1)
    {
        // ====== Broadcast mode ======
        // Root provides the message; all ranks receive the same text.
        std::string received = bcast_string(args.msg, 0, MPI_COMM_WORLD);

        for (int r = 0; r < shared_repeat; ++r)
        {
            std::cout << "[bcast] rank " << rank << "/" << world
                      << " received: " << received << "\n";
        }
        std::cout.flush();
    }
    else
    {
        // ====== Point-to-point mode ======
        // Protocol: root sends (len, bytes) to each worker; workers receive and print.
        if (rank == 0)
        {
            const int len = static_cast<int>(args.msg.size());
            for (int dest = 1; dest < world; ++dest)
            {
                for (int r = 0; r < shared_repeat; ++r)
                {
                    MPI_Send(&len, 1, MPI_INT, dest, shared_tag, MPI_COMM_WORLD);
                    if (len > 0)
                    {
                        MPI_Send(args.msg.data(), len, MPI_CHAR, dest, shared_tag, MPI_COMM_WORLD);
                    }
                }
            }
            // Optional: root prints a small confirmation
            std::cout << "[pt2pt] rank 0/" << world
                      << " sent message to ranks 1.." << (world - 1)
                      << " tag=" << shared_tag
                      << " repeat=" << shared_repeat << "\n";
            std::cout.flush();
        }
        else
        {
            for (int r = 0; r < shared_repeat; ++r)
            {
                int len = 0;
                MPI_Recv(&len, 1, MPI_INT, 0, shared_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::string msg;
                msg.resize(static_cast<size_t>(len));
                if (len > 0)
                {
                    MPI_Recv(msg.data(), len, MPI_CHAR, 0, shared_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                std::cout << "[pt2pt] rank " << rank << "/" << world
                          << " received: " << msg << "\n";
            }
            std::cout.flush();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // neat exit
    if (rank == 0)
    {
        std::cerr.flush();
        std::cout << "Done.\n";
    }
    MPI_Finalize();
    return 0;
}
