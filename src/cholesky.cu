// #include <cublas_v2.h>
#include <iostream>
#include <cusolverDn.h>
using namespace std;


// void tc_rpotrf(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle, int n, float *A, int lda, float *work, __half *hwork, int nb, int *devInfo)
// {
//     if(n <= nb)
//     {
//         cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
//                         nb, A, lda, work, lwork, devInfo);
//         return;
//     }


//     tc_rpotrf(cublas_handle, cusolver_handle, n/2, A, lda, work, hwork, nb, devInfo);

//     tc_rtrsm(cublas_handle, n/2, n/2, A, lda, A+n/2, lda, hwork, trsm_nb);
    
//     tc_syrk(cublas_handle, n/2, n/2, A+n/2, lda, A+n/2+n/2*lda, lda, hwork, syrk_nb);

//     tc_rpotrf(cublas_handle, cusolver_handle, n/2, A+n/2+n/2*lda, lda, work, hwork, nb, devInfo);

//     return;
// }


int rec_cholesky(cublasHandle_t cublas_handle,cusolverDnHandle_t cusolver_handle, double *A, long ldA,int n, int *devInfo)
{
    // 0.判断是否需要调用cuSolver
    if(n <= 64)
    {
        int Lwork;
        
        cusolverDnDpotrf_bufferSize( cusolver_handle,
                 CUBLAS_FILL_MODE_LOWER,
                 n,
                 A,
                 ldA,
                 &Lwork);

        double *work;
        cudaMalloc((void**)&work, sizeof(double)*Lwork);

        cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, 
        n, A, ldA, work, Lwork, devInfo);

        cudaFree(work);

        return 0;

    }
    
    // 1.把A分解为A11、A12、A21、A22
    int n2 = n/2;

    // 2.对A11进行Cholesky分解,求出L11
    double *A11 = A;
    int ldA11 = ldA;
    rec_cholesky(cublas_handle,cusolver_handle, A11, ldA11, n2, devInfo);

    double dOne = 1.0;

    // 3.对A21进行trsm,求出L21 = L11^-1 * A21
    double *A21 = A + n2;
    cublasDtrsm(cublas_handle,
                            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                           n2, n2,
                           &dOne,
                           A11, ldA,
                           A21, ldA);

    // 4.对A22 - L21 * L21^T进行syrk
    double dNegOne = -1.0;
    double *A22 = A + n2 + n2*ldA;
    cublasDsyrk(cublas_handle,
                            CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N,
                            n2, n2,
                            &dNegOne,
                            A21, ldA,
                            &dOne,
                            A22, ldA);

    // 5.递归调用此函数，对A22 - L21 * L21^T进行Cholesky分解
    rec_cholesky(cublas_handle, cusolver_handle, A22, ldA, n2, devInfo);

    return 0;
}


int main(int argc, char *argv[])
{
    if (2 != argc)
    {
        cout << "Usage(b = nb in ZY): AppName <n>" << endl;
        return 0;
    }

    int m, n;
    m = n = atol(argv[1]);


    // 1.初始化cuBLAS和cuSolver
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    // 2.初始化矩阵A
    // int n = 1024;
    double *A;
    
    cudaMalloc((void**)&A, sizeof(double)*n*n);

    // generateUniformMatrix(A, n, n);
    // dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    // dim3 blockDim(32, 32);
    // launchKernel_CpyMatrixL2U(gridDim, blockDim, n, A, n);

    // 3.调用递归函数rec_cholesky
    int devInfo;
    cudaMalloc((void**)&devInfo, sizeof(int));

    rec_cholesky(cublas_handle, cusolver_handle, A, n, n, &devInfo);

    // 4.释放cuBLAS和cuSolver
    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);

    // 5.释放矩阵A
    cudaFree(A);

    return 0;
}

