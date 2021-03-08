#include <cusparse.h>
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
class DevSparseMat
{
public:

};


template<typename Scalar>
DevSparseMat<Scalar>
spdiags(cv::Mat_<Scalar> &_Data, int m, int n)
{

}


template<typename Scalar>
void gpu_sparse_solve(DevSparseMat<Scalar> &A, const cv::Mat &B, cv::Mat &X)
/*
 * Solve AX = B for X
 * where A is sparse, X and B are vectors
 */
{
	auto ctx = CuSparseSolver::getInstance().get();

	// prepare structs
	cusparseMatDescr_t descrA;
	cusparseCreateMatDescr(&descrA);

//	cusparseScsrsv2_solve
}
