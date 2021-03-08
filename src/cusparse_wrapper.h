#include <cusparse.h>


template<typename Scalar>
class DevSparseMat
{
public:

};


template<typename Scalar>
void spdiags(cv::Mat_<Scalar> &_Data, int m, int n)
{

}


template<typename Scalar>
void gpu_sparse_solve(DevSparseMat<Scalar> &A, const cv::Mat &B, cv::Mat &X)
/*
 * Solve AX = B for X
 * where A is sparse, X and B are vectors
 */
{

}
