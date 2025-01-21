#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <time.h>


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__, cudaGetErrorString(e));  \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__, ncclGetErrorString(r));  \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


int main(int argc, char* argv[])
{
  int size = 256 * 1024 * 1024;

  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  //calculating localRank which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }


  //each process is using two GPUs
  if (argc < 2) {
    printf("Usage: %s <ngpu_per_proc>\n", argv[0]);
    return 1;
  }
  int nDev = std::atoi(argv[1]);

  float** hbuff = (float**)malloc(nDev * sizeof(float*));
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDev);

  //picking GPUs based on localRank
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(localRank * nDev + i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  for (int i = 0; i < nDev; ++i) {
    hbuff[i] = (float*)malloc(size * sizeof(float));
    for (int j = 0; j < size; ++j)
      hbuff[i][j] = 1.0f;
    CUDACHECK(cudaMemcpy(sendbuff[i], hbuff[i], size * sizeof(float), cudaMemcpyHostToDevice));
  }


  // xx1,xx2 is for three-phase allreduce,
  // xx1 for within pcie-sw, xx2 for across pcie-sw
  ncclUniqueId id, id1[2], id2[4];
  ncclComm_t comms[nDev], comms1[nDev], comms2[nDev];


  //generating NCCL unique ID at one process and broadcasting it to all
  if (myRank == 0) {
      ncclGetUniqueId(&id);
      for (int i=0; i<2; ++i)
          ncclGetUniqueId(id1+i);
      for (int i=0; i<4; ++i)
          ncclGetUniqueId(id2+i);
  }
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Bcast((void *)id1, 2 * sizeof(id1[0]), MPI_BYTE, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Bcast((void *)id2, 4 * sizeof(id2[0]), MPI_BYTE, 0, MPI_COMM_WORLD));


  //initializing NCCL, group API is required around ncclCommInitRank as it is
  //called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  int nGpus = nRanks * nDev;
  int offGpus = myRank * nDev;
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaSetDevice(localRank * nDev + i));
     NCCLCHECK(ncclCommInitRank(comms+i, nGpus, id, offGpus + i));
#if 1
     NCCLCHECK(ncclCommInitRank(comms1+i, nGpus / 2, id1[(offGpus + i) / 4], (offGpus + i) % 4));
     NCCLCHECK(ncclCommInitRank(comms2+i, nGpus / 4, id2[(offGpus + i) % 4], (offGpus + i) / 4));
#else
     NCCLCHECK(ncclCommInitRank(comms1+i, nGpus / 2, id1[(offGpus + i) % 2], (offGpus + i) / 2));
     NCCLCHECK(ncclCommInitRank(comms2+i, nGpus / 4, id2[(offGpus + i) / 2], (offGpus + i) % 2));
#endif
  }
  NCCLCHECK(ncclGroupEnd());


  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread/process

  clock_t start, end;
  double wall_time_elapsed = 0.0f;
  int warmup = 10;
  int niters = 10;
  if (argc >= 3)
    warmup = std::atoi(argv[2]);
  if (argc >= 4)
    niters = std::atoi(argv[3]);
  for (int n=0; n<warmup + niters; ++n) {
      if (n == warmup)
          start = clock();
#if 0
      NCCLCHECK(ncclGroupStart());
      for (int i=0; i<nDev; i++)
          NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
              comms[i], s[i]));
      NCCLCHECK(ncclGroupEnd());
#else
      // reduce scatter within pcie-switch
///*
      NCCLCHECK(ncclGroupStart());
      for (int i=0; i<nDev; i++)
          NCCLCHECK(ncclReduceScatter((const void*)sendbuff[i], (void*)recvbuff[i], size / 4, ncclFloat, ncclSum,
              comms1[i], s[i]));
      NCCLCHECK(ncclGroupEnd());
//*/
      // all-reduce across pcie-switch
///*
      NCCLCHECK(ncclGroupStart());
      for (int i=0; i<nDev; i++) {
#if 1
          NCCLCHECK(ncclAllReduce((const void*)recvbuff[i], (void*)sendbuff[i], size / 4, ncclFloat, ncclSum,
              comms2[i], s[i]));
#else
          NCCLCHECK(ncclSend((const void*)sendbuff[i], size / 4, ncclFloat, ((offGpus + i) / 4  + 1) % 2,
                comms2[i], s[i]));
          NCCLCHECK(ncclRecv((void*)recvbuff[i], size / 4, ncclFloat, ((offGpus + i) / 4 + 1) % 2,
                comms2[i], s[i]));
#endif
      }
      NCCLCHECK(ncclGroupEnd());
//*/

      // all-gather within pcie-switch
///*
      NCCLCHECK(ncclGroupStart());
      for (int i=0; i<nDev; i++)
          NCCLCHECK(ncclAllGather((const void*)sendbuff[i], (void*)recvbuff[i], size / 4, ncclFloat,
              comms1[i], s[i]));
      NCCLCHECK(ncclGroupEnd());
//*/
#endif
      //synchronizing on CUDA stream to complete NCCL communication
      for (int i=0; i<nDev; i++)
          CUDACHECK(cudaStreamSynchronize(s[i]));
  }
  end = clock();
  wall_time_elapsed = ((double) (end - start)) / niters / CLOCKS_PER_SEC;
  printf("Time elapsed: %lf seconds per iter, count %d, and avg bw: %lf \n", wall_time_elapsed, size, (double)(size) * sizeof(ncclFloat) * 2 * (nGpus - 1) / nGpus / wall_time_elapsed);

  if (warmup == 0 && niters == 1) {
    for (int i=0; i<nDev; i++) {
      CUDACHECK(cudaMemcpy(hbuff[i], recvbuff[i], size * sizeof(ncclFloat), cudaMemcpyDeviceToHost));
      for (int j = 0; j < size; ++j)
        if (hbuff[i][j] != (float)nGpus) {
          printf("Wrong ! [%d] host(%d): %lf (vs %d)\n", i, j, hbuff[i][j], nGpus);
          return 1;
        }
    }
    printf("Correct! The results are.\n");
  }


  //freeing device memory
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
  }


  //finalizing NCCL
  for (int i=0; i<nDev; i++) {
     ncclCommDestroy(comms[i]);
     ncclCommDestroy(comms1[i]);
     ncclCommDestroy(comms2[i]);
  }

  //freeing host memory
  for (int i=0; i<nDev; i++) {
     free(hbuff[i]);
  }

  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
