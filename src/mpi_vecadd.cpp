// SIT315 M3 Activity 02 — Distributed Vector Addition (no hardcoding; flag-driven)
// Modes:
//   --mode=scatter  : MPI_Scatterv + local add + MPI_Gatherv
//   --mode=pt2pt    : manual MPI_Send / MPI_Recv distribution + collection
// Flags:
//   --n=1000000     : vector length (default 100000)
//   --seed=123      : RNG seed (default 123)
//   --repeat=1      : extra compute repeats for timing (>=1)
//   --check=1       : verify result on root (default 1)
//   --print=5       : on root, print first K results (default 0)
//   --help          : usage
//
// Build on node1:
//   mpic++ -O2 -std=c++17 src/mpi_vecadd.cpp -o build/mpi_vecadd -Wall -Wextra -Wpedantic
//
// Example runs (from node1):
//   mpirun -np 4 ./build/mpi_vecadd --mode=scatter --n=1000000 --check=1
//   mpirun -np 4 ./build/mpi_vecadd --mode=pt2pt   --n=1000000 --check=1 --repeat=3

#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Args {
    std::string mode = "scatter";
    std::int64_t n   = 100000;   // default length
    int seed         = 123;
    int repeat       = 1;
    int check        = 1;
    int print_k      = 0;
    bool help        = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string t(argv[i]);
        if (t == "--help" || t == "-h") { a.help = true; continue; }
        auto eq = t.find('=');
        if (eq == std::string::npos) continue;
        auto key = t.substr(0, eq);
        auto val = t.substr(eq + 1);
        if (key == "--mode")   a.mode    = val;
        if (key == "--n")      a.n       = std::stoll(val);
        if (key == "--seed")   a.seed    = std::stoi(val);
        if (key == "--repeat") a.repeat  = std::max(1, std::stoi(val));
        if (key == "--check")  a.check   = std::stoi(val);
        if (key == "--print")  a.print_k = std::stoi(val);
    }
    return a;
}

static void make_counts_displs(std::int64_t N, int world,
                               std::vector<int> &counts,
                               std::vector<int> &displs) {
    counts.assign(world, 0);
    displs.assign(world, 0);
    if (N < 0) N = 0;
    std::int64_t base = (world > 0) ? (N / world) : 0;
    std::int64_t rem  = (world > 0) ? (N % world) : 0;
    int off = 0;
    for (int r = 0; r < world; ++r) {
        std::int64_t c = base + (r < rem ? 1 : 0);
        counts[r] = static_cast<int>(c);   // clamp to int for MPI
        displs[r] = off;
        off += counts[r];
    }
}

static void print_sample(const std::vector<int> &A,
                         const std::vector<int> &B,
                         const std::vector<int> &C,
                         int k) {
    k = std::min<int>(k, std::min((int)A.size(), std::min((int)B.size(), (int)C.size())));
    for (int i = 0; i < k; ++i) {
        std::cout << "C[" << i << "] = A[" << i << "] + B[" << i
                  << "] -> " << A[i] << " + " << B[i] << " = " << C[i] << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    Args args = parse_args(argc, argv);
    if (args.help) {
        if (rank == 0) {
            std::cout <<
"Usage: mpirun -np P ./mpi_vecadd --mode=<scatter|pt2pt> [--n=N] [--seed=S] [--repeat=R] [--check=0|1] [--print=K]\n";
        }
        MPI_Finalize();
        return 0;
    }
    if (args.n < 0) args.n = 0;

    // Root allocates full A,B (int vectors). Others allocate nothing yet.
    std::vector<int> A, B, C; // only fully sized on root
    if (rank == 0) {
        A.resize(args.n);
        B.resize(args.n);
        std::mt19937 rng(args.seed);
        std::uniform_int_distribution<int> dist(0, 99);
        for (std::int64_t i = 0; i < args.n; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }
    }

    // Compute counts and displacements for general N
    std::vector<int> counts, displs;
    make_counts_displs(args.n, world, counts, displs);
    int local_n = counts[rank];

    std::vector<int> a_local(local_n), b_local(local_n), c_local(local_n);

    double t0 = 0.0, t1 = 0.0;
    if (args.mode == "scatter") {
        // ===== Strategy 1: collectives (Scatterv/Gatherv) =====
        t0 = MPI_Wtime();

        // Distribute
        MPI_Scatterv(A.data(), counts.data(), displs.data(), MPI_INT,
                     a_local.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(B.data(), counts.data(), displs.data(), MPI_INT,
                     b_local.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

        // Compute (optionally repeat for timing stability)
        for (int r = 0; r < args.repeat; ++r) {
            for (int i = 0; i < local_n; ++i) c_local[i] = a_local[i] + b_local[i];
        }

        // Collect
        if (rank == 0) C.resize(args.n);
        MPI_Gatherv(c_local.data(), local_n, MPI_INT,
                    C.data(), counts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        t1 = MPI_Wtime();
        if (rank == 0) {
            std::cout << "[scatter] n=" << args.n
                      << " world=" << world
                      << " repeat=" << args.repeat
                      << " time_s=" << std::fixed << std::setprecision(6) << (t1 - t0)
                      << "\n";
        }
    } else if (args.mode == "pt2pt") {
        // ===== Strategy 2: manual point-to-point =====
        t0 = MPI_Wtime();

        if (rank == 0) {
            // Send sizes and chunks to each worker
            for (int r = 1; r < world; ++r) {
                int cnt = counts[r];
                MPI_Send(&cnt, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                if (cnt > 0) {
                    MPI_Send(A.data() + displs[r], cnt, MPI_INT, r, 1, MPI_COMM_WORLD);
                    MPI_Send(B.data() + displs[r], cnt, MPI_INT, r, 2, MPI_COMM_WORLD);
                }
            }
            // Root's own local data
            std::copy(A.begin() + displs[0], A.begin() + displs[0] + counts[0], a_local.begin());
            std::copy(B.begin() + displs[0], B.begin() + displs[0] + counts[0], b_local.begin());
        } else {
            int cnt = 0;
            MPI_Recv(&cnt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (cnt != local_n) {
                a_local.assign(cnt, 0);
                b_local.assign(cnt, 0);
                c_local.assign(cnt, 0);
                local_n = cnt;
            }
            if (cnt > 0) {
                MPI_Recv(a_local.data(), cnt, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(b_local.data(), cnt, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Compute
        for (int r = 0; r < args.repeat; ++r) {
            for (int i = 0; i < local_n; ++i) c_local[i] = a_local[i] + b_local[i];
        }

        // Gather manually to root
        if (rank == 0) {
            C.assign(args.n, 0);
            std::copy(c_local.begin(), c_local.end(), C.begin() + displs[0]);
            for (int r = 1; r < world; ++r) {
                int cnt = 0;
                MPI_Recv(&cnt, 1, MPI_INT, r, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (cnt > 0) {
                    MPI_Recv(C.data() + displs[r], cnt, MPI_INT, r, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            MPI_Send(&local_n, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            if (local_n > 0) {
                MPI_Send(c_local.data(), local_n, MPI_INT, 0, 4, MPI_COMM_WORLD);
            }
        }

        t1 = MPI_Wtime();
        if (rank == 0) {
            std::cout << "[pt2pt] n=" << args.n
                      << " world=" << world
                      << " repeat=" << args.repeat
                      << " time_s=" << std::fixed << std::setprecision(6) << (t1 - t0)
                      << "\n";
        }
    } else {
        if (rank == 0) std::cerr << "Error: --mode must be 'scatter' or 'pt2pt'\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Optional verification on root
    if (args.check && rank == 0) {
        std::size_t mism = 0;
        for (std::int64_t i = 0; i < args.n; ++i) {
            int expect = A[i] + B[i];
            if (C[i] != expect) { ++mism; if (mism < 5) {
                std::cerr << "[mismatch] i=" << i << " got=" << C[i] << " expect=" << expect << "\n";
            }}
        }
        std::cout << "[check] mismatches=" << mism << "\n";
        if (args.print_k > 0) {
            print_sample(A, B, C, args.print_k);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Done.\n";
    MPI_Finalize();
    return 0;
}
