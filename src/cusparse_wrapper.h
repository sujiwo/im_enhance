#include <cusparse.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Sparse>
#include <opencv2/core.hpp>


/*
 * Singleton for cusparse
 */
class CuSparseSolver
{
public:
	static CuSparseSolver& getInstance()
	{
		static CuSparseSolver instance;
		return instance;
	}

	cusparseHandle_t &get() { return handle; }

private:
	CuSparseSolver()
	{
		cusparseCreate(&handle);
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
	}

    cusparseHandle_t handle;

public:
    CuSparseSolver(CuSparseSolver const&)  = delete;
    void operator=(CuSparseSolver const&)  = delete;
};


template<typename Scalar>
using SparseMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;


template<typename Scalar>
class DevSparseMat : public SparseMat<Scalar>
{
public:

static DevSparseMat<Scalar>
spdiags(cv::Mat_<Scalar> &_Data, int rows, int cols)
{

}
};


typedef Eigen::SparseMatrix<float, Eigen::RowMajor> Spf;
void gpu_sparse_solve(Spf &A, const cv::Mat &B, cv::Mat &X)
/*
 * Solve AX = B for X
 * where A is sparse, X and B are vectors
 *
 * A must be in CSR format.
 */
{
	auto ctx = CuSparseSolver::getInstance().get();

	// prepare structs
	cusparseMatDescr_t descrA;
	csrsv2Info_t info;
	int pBufferSize;
	void *pBuffer;

	cusparseCreateMatDescr(&descrA);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

	cusparseCreateCsrsv2Info(&info);

	cusparseScsrsv2_bufferSize(ctx, CUSPARSE_OPERATION_NON_TRANSPOSE, A.rows(), A.nonZeros(), descrA, A.valuePtr(), A.outerIndexPtr(), A.innerIndexPtr(), info, &pBufferSize);
	cudaMalloc((void**)&pBuffer, pBufferSize);
	pBuffer = malloc(pBufferSize);
	cusparseScsrsv2_analysis(ctx, CUSPARSE_OPERATION_NON_TRANSPOSE, A.rows(), A.nonZeros(), descrA, A.valuePtr(), A.outerIndexPtr(), A.innerIndexPtr(), info,  CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);
//	cusparseScsrsv2_solve

	cudaFree(pBuffer);
	cusparseDestroyCsrsv2Info(info);
}
