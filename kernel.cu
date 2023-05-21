#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cub/cub.cuh>


#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X, Y) ((X) > (Y) ? (Y) : (X))
#define ABS(X) ((X) < 0 ? -1 * (X) : (X))

#define gpuErrchk(ans, A, Anew, A_d, max) { gpuAssert((ans), __FILE__, __LINE__, A, Anew, A_d, max); }
inline void gpuAssert(cudaError_t code, const char* file, int line, double* A, double* Anew, double* A_d, double* max, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        free(A);
        if (Anew != NULL) cudaFree(&Anew); 
        if (A_d != NULL) cudaFree(&A_d);  
        if (max != NULL) cudaFree(&max);
        if (abort) exit(code);
    }
}

__global__ void initMatrix(
    double* A, double* Anew,
    int netSize, double hst, double hsb, double vsl, double vsr,
    double tl, double tr, double bl, double br)
{
    //this functions initializes matrix borders in O(n)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= netSize*netSize) return;
    A[i * netSize] = vsl * i + tl;
    A[i] = hst * i + tl;
    A[((netSize - 1) - i) * netSize + (netSize - 1)] = vsr * i + br;
    A[(netSize - 1) * netSize + ((netSize - 1) - i)] = hsb * i + br;

    Anew[i * netSize] = vsl * i + tl;
    Anew[i] = hst * i + tl;
    Anew[((netSize - 1) - i) * netSize + (netSize - 1)] = vsr * i + br;
    Anew[(netSize - 1) * netSize + ((netSize - 1) - i)] = hsb * i + br;
}

void printMatrix(double* A, int netSize)
{
    std::cout << "netSize: " << netSize << std::endl;
    for (int i = 0; i < netSize; i++)
    {
        for (int j = 0; j < netSize; j++)
        {
            std::cout << A[i * netSize + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

//finds mean of 4 neighbours. Doesn't recalculate borders
__global__ void iterateMatrix(double* A, double* Anew, int netSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (netSize) || i > netSize * (netSize - 1) || i % netSize == 0 || i % netSize == netSize - 1) return;
    Anew[i] = 0.25 * (A[i - netSize] + A[i + netSize] + A[i - 1] + A[i + 1]);
}

//applied to two matrices to find difference. Writes result to a new array
__global__ void findDifference(double* A, double* Anew, double* result, int netSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= netSize*netSize) return;

    result[i] = ABS(Anew[i] - A[i]);
}

//swaps pointers on device
void swapMatrix(double* &A, double* &Anew)
{
    double* temp = A;
    A = Anew;
    Anew = temp;
}

int main(int argc, char* argv[])
{
    int threads = 512; //for threadsPerBlock variable

    int netSize = 128;
    double minError = 0.000001;
    int maxIterations = 0;
    int toPrintResult = 0;
    char* end;

    //correct input check
    if (argc != 5) {
        std::cout << "You must enter exactly 4 arguments:\n1. Grid size (one number)\n2. Minimal error\n3. Iterations amount\n4. Print final result (1 - Yes/0 - No)\n";
        return -1;
    }
    else {
        netSize = strtol(argv[1], &end, 10);
        minError = strtod(argv[2], &end);
        maxIterations = strtol(argv[3], &end, 10);
        toPrintResult = strtol(argv[4], &end, 10);
    }
    std::cout << netSize << " " << minError << " " << maxIterations << std::endl;

    dim3 threadsPerBlock = dim3(threads);

    //values of net edges
    const double tl = 10, //top left
        tr = 20, //top right
        bl = 20, //bottom left
        br = 30; //bottom right

    const double hst = (tr - tl) / (netSize - 1), //horizontal step top
        hsb = (bl - br) / (netSize - 1), //horizontal step bottom
        vsl = (bl - tl) / (netSize - 1), //vertical step left
        vsr = (tr - br) / (netSize - 1); //vertical step right

    double* A_h = (double*)malloc(sizeof(double*) * netSize * netSize);
    double* A_d =NULL;
    double* Anew = NULL;
    double* max = NULL; //actually it meant to be d_diff, but I can't rename it...
    double* d_error = NULL;

    //allocating memory to A_d, Anew, max
    gpuErrchk( cudaMalloc(&A_d, sizeof(double*) * netSize * netSize), A_h, Anew, A_d, max);
    gpuErrchk( cudaMalloc(&Anew, sizeof(double*) * netSize * netSize), A_h, Anew, A_d, max );
    gpuErrchk( cudaMalloc(&max, sizeof(double*) * netSize * netSize), A_h, Anew, A_d, max );
    gpuErrchk(cudaMalloc(&d_error, sizeof(double)), A_h, Anew, A_d, max);

    //setting A_d matrix to zero's
    gpuErrchk( cudaMemset(A_d, 0, sizeof(double) * netSize), A_h, Anew, A_d, max );

    //finding size of memory need to be allocated to d_tempStorage
    void* d_tempStorage = NULL;
    size_t d_tempStorageBytes = 0;
    cub::DeviceReduce::Max(d_tempStorage, d_tempStorageBytes, max, d_error, netSize*netSize);

    //allocating memory to d_tempStorage
    gpuErrchk( cudaMalloc(&d_tempStorage, d_tempStorageBytes), A_h, Anew, A_d, max);

    //initialising A_d and Anew (device)
    initMatrix <<< MAX((int)(netSize / threadsPerBlock.x), 1), MIN(threadsPerBlock.x, netSize) >>> (A_d, Anew, netSize, hst, hsb, vsl, vsr, tl, tr, bl, br);
    gpuErrchk( cudaGetLastError(), A_h, Anew, A_d, max );


    double error = 10.;
    int iteration = 0;
    int xBonus = netSize % threadsPerBlock.x == 0 ? 0 : 1; //additional block if not enough threads
    dim3 blockNum = dim3(MAX((int)((netSize * netSize) / threadsPerBlock.x), 1) + xBonus);

    //we want to get cuda graph out of kernels
    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t graphInstance;
    cudaStream_t stream;

    printf("init done!\n");

    //main loop
    while (error > minError && iteration < maxIterations)
    {

        if(!graphCreated) {
            printf("Preparing for the first launch...\n");
            gpuErrchk( cudaStreamCreate(&stream), A_h, Anew, A_d, max );
            gpuErrchk( cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), A_h, Anew, A_d, max );

            //do 100 at time
            for (int i = 0; i < 100; ++i) {
                iterateMatrix <<< blockNum, threadsPerBlock, 0, stream >>> ( A_d, Anew, netSize );
                swapMatrix(A_d, Anew);
            }

            gpuErrchk( cudaStreamEndCapture(stream, &graph), A_h, Anew, A_d, max );
            gpuErrchk( cudaGraphInstantiate(&graphInstance, graph, NULL, NULL, 0), A_h, Anew, A_d, max );
            printf("graph instantiated!\n");
            graphCreated = true;
        }

        gpuErrchk( cudaGraphLaunch(graphInstance, stream), A_h, Anew, A_d, max );
        gpuErrchk( cudaStreamSynchronize(stream), A_h, Anew, A_d, max );

        //get an error if happened
        gpuErrchk( cudaGetLastError(), A_h, Anew, A_d, max );

        //every 100 iteration will be documented 
        //finding max error (max difference element from matrix A_d and Anew)
        findDifference <<< blockNum, threadsPerBlock >>> (A_d, Anew, max, netSize);
        cub::DeviceReduce::Max(d_tempStorage, d_tempStorageBytes, max, d_error, netSize * netSize);

        //copy error to CPU
        gpuErrchk( cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost), A_h, Anew, A_d, max );

        //print report for this iteration
        std::cout << "iteration " << iteration + 1 << "/" << maxIterations << " error = " << error << "\t(min " << minError << ")" << std::endl;

        //check for errors
        gpuErrchk( cudaGetLastError(), A_h, Anew, A_d, max );
        iteration += 100;
    }

    if (toPrintResult) {
        //print matrix
        gpuErrchk(cudaMemcpy(A_h, Anew, sizeof(double) * netSize * netSize, cudaMemcpyDeviceToHost), A_h, Anew, A_d, max);
        gpuErrchk(cudaDeviceSynchronize(), A_h, Anew, A_d, max)
        printMatrix(A_h, netSize);
    }

    //find final error
    findDifference <<< blockNum, threadsPerBlock >>> (A_d, Anew, max, netSize);
    cub::DeviceReduce::Max(d_tempStorage, d_tempStorageBytes, max, d_error, netSize * netSize);

    //copy error to CPU
    gpuErrchk(cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost), A_h, Anew, A_d, max);

    //print final report
    std::cout << "Program ended on iteration " << iteration << ". Final error = " << error << "\t(min " << minError << ")" << std::endl;

    //check for errors
    gpuErrchk(cudaGetLastError(), A_h, Anew, A_d, max);

    //free memory
    cudaFree(&A_d);
    cudaFree(&Anew);
    cudaFree(&max); cudaFree(&d_error); cudaFree(&d_tempStorage);

    return 0;
}
