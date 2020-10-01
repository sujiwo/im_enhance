// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 <darcy.beurle@ibnm.uni-hannover.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MUMPSSUPPORT_H
#define EIGEN_MUMPSSUPPORT_H

#include <complex>

#include <stdexcept>

namespace Eigen
{
#if defined(DCOMPLEX)
#define MUMPS_COMPLEX COMPLEX
#define MUMPS_DCOMPLEX DCOMPLEX
#else
#define MUMPS_COMPLEX std::complex<float>
#define MUMPS_DCOMPLEX std::complex<double>
#endif

template <typename _MatrixType>
class MUMPSLU;
template <typename _MatrixType, int Options>
class MUMPSLDLT;

namespace internal
{
template <class Mumps>
struct mumps_traits;

template <typename _MatrixType>
struct mumps_traits<MUMPSLU<_MatrixType>>
{
	typedef _MatrixType MatrixType;
	typedef typename _MatrixType::Scalar Scalar;
	typedef typename _MatrixType::RealScalar RealScalar;
	typedef typename _MatrixType::StorageIndex StorageIndex;
};

template <typename _MatrixType, int Options>
struct mumps_traits<MUMPSLDLT<_MatrixType, Options>>
{
	typedef _MatrixType MatrixType;
	typedef typename _MatrixType::Scalar Scalar;
	typedef typename _MatrixType::RealScalar RealScalar;
	typedef typename _MatrixType::StorageIndex StorageIndex;
};

template <typename T>
struct MUMPSAPIWrapper;

template <>
struct MUMPSAPIWrapper<float>
{
	using MUMPS_STRUC_C = SMUMPS_STRUC_C;
	static void mumps_c(MUMPS_STRUC_C& info) { smumps_c(&info); }
};

template <>
struct MUMPSAPIWrapper<double>
{
	using MUMPS_STRUC_C = DMUMPS_STRUC_C;
	static void mumps_c(MUMPS_STRUC_C& info) { dmumps_c(&info); }
};

template <>
struct MUMPSAPIWrapper<std::complex<float>>
{
	using MUMPS_STRUC_C = CMUMPS_STRUC_C;
	static void mumps_c(MUMPS_STRUC_C& info) { cmumps_c(&info); }
};

template <>
struct MUMPSAPIWrapper<std::complex<double>>
{
	using MUMPS_STRUC_C = ZMUMPS_STRUC_C;
	static void mumps_c(MUMPS_STRUC_C& info) { zmumps_c(&info); }
};

namespace mumps
{
enum ordering { AMD, AMF = 2, Scotch, Pord, Metis, QAMD, automatic };

// Jobs in MUMPS use the following:
// 4   Job = 1 && Job = 2
// 5   Job = 2 && Job = 3
// 6   Job = 1 && Job = 2 && Job = 3
enum job {
	terminate = -2,
	initialisation = -1,
	analysis = 1,
	factorisation = 2,
	back_substitution = 3
};

enum residual { none, expensive, cheap };

enum matrix_property { unsymmetric, SPD, general_symmetric };
}
}

// Base class to interface with MUMPS. Users should not instantiate this class directly.
template <class Derived>
class MumpsBase : public SparseSolverBase<Derived>
{
protected:
	typedef SparseSolverBase<Derived> Base;
	using Base::derived;
	using Base::m_isInitialized;

public:
	using Base::_solve_impl;

	typedef typename internal::mumps_traits<Derived>::MatrixType _MatrixType;
	typedef _MatrixType MatrixType;
	typedef typename MatrixType::Scalar Scalar;
	typedef typename MatrixType::RealScalar RealScalar;
	typedef typename MatrixType::StorageIndex StorageIndex;
	typedef Matrix<Scalar, Dynamic, 1> Vector;
	typedef SparseMatrix<Scalar, ColMajor> ColSpMatrix;
	enum {
		ColsAtCompileTime = MatrixType::ColsAtCompileTime,
		MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
	};

	using solver_api_type = internal::MUMPSAPIWrapper<Scalar>;

public:
	MumpsBase() : m_initisOk(false), m_analysisIsOk(false), m_factorizationIsOk(false), m_size(0) {}

	~MumpsBase()
	{
		m_solver.job = internal::mumps::job::terminate;
		solver_api_type::mumps_c(m_solver);
	}

	/* Solve the system */
	template <typename Rhs, typename Dest>
	bool _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const;

	Index cols() const { return m_size; }
	Index rows() const { return m_size; }

	ComputationInfo info() const
	{
		eigen_assert(m_isInitialized && "Decomposition is not initialized.");
		return m_info;
	}

protected:
	// Initialize the MUMPS data structure, check the matrix
	void init(bool const is_matrix_symmetric);

	// Compute the ordering and the symbolic factorization
	void analyzePattern();

	// Compute the numerical factorization
	void factorize();

	void compute();

protected:
	int m_initisOk;
	int m_analysisIsOk;
	int m_factorizationIsOk;
	mutable ComputationInfo m_info;

	typename internal::MUMPSAPIWrapper<Scalar>::MUMPS_STRUC_C mutable m_solver;

	mutable MUMPS_INT m_size; // Size of the matrix

	mutable Array<MUMPS_INT, Dynamic, 1> m_rows, m_cols; // Coordinate format
	mutable Vector m_coeffs;                             // Coefficients in sparse matrix
};

template <class Derived>
void MumpsBase<Derived>::init(bool const is_matrix_symmetric)
{
	m_solver.job = internal::mumps::job::initialisation;

	// Par determines parallelisation of the computation
	// 0  : Host is not involved with computation
	// 1  : Host is involved
	m_solver.par = 1;

	// Sym is 1 if the matrix is symmetric (triangular) and 0 if the matrix
	// is full.  This determines if LDLT or LU factorisation is used
	// respectively
	m_solver.sym = is_matrix_symmetric;

	// No distributed memory MPI Communicator for horrible FORTRAN compatibility
	m_solver.comm_fortran = -987654;

	solver_api_type::mumps_c(m_solver);

	std::cout << "Initialisation complete!\nSet verbosity levels\n";

	// Verbosity levels (six is lots of noise)
	m_solver.icntl[0] = 6;
	m_solver.icntl[1] = 6;
	m_solver.icntl[2] = 6;
	m_solver.icntl[3] = 6;

	// Ordering algorithm
	m_solver.icntl[6] = internal::mumps::ordering::automatic;

	// Iterative refinement limit
	m_solver.icntl[9] = 100;

	// Compute the residual
	m_solver.icntl[10] = internal::mumps::residual::cheap;

	m_size = 0;

	// Check the returned error
	if (m_solver.info[0] < 0)
	{
		m_info = InvalidInput;
		m_initisOk = false;
		throw std::domain_error("Error code " + std::to_string(m_solver.info[0])
		+ " in MUMPS initialization");
	}
	else
	{
		m_info = Success;
		m_initisOk = true;
	}
}

template <class Derived>
void MumpsBase<Derived>::compute()
{
	eigen_assert(m_rows.size() == m_cols.size() && "The input matrix should be square");
	analyzePattern();
	factorize();
}

template <class Derived>
void MumpsBase<Derived>::analyzePattern()
{
	eigen_assert(m_initisOk && "The initialization of MUMPS failed");

	std::cout << "analyzePattern size " << m_size << std::endl;

	m_solver.job = internal::mumps::job::analysis;

	m_solver.n = m_size;
	m_solver.nz = internal::convert_index<MUMPS_INT8>(m_cols.size());

	m_solver.a = 0;
	m_solver.rhs = 0;

	m_solver.irn = m_rows.data();
	m_solver.jcn = m_cols.data();

	solver_api_type::mumps_c(m_solver);

	// Check the returned error
	if (m_solver.info[0] < 0)
	{
		m_info = NumericalIssue;
		m_analysisIsOk = false;
		throw std::domain_error("Error code " + std::to_string(m_solver.info[0])
		+ " in MUMPS analyzePattern()");
	}
	else
	{
		m_info = Success;
		m_analysisIsOk = true;
	}
}

template <class Derived>
void MumpsBase<Derived>::factorize()
{
	eigen_assert(m_analysisIsOk && "analysePattern() should be called before factorize()");
	eigen_assert(m_cols.size() == m_rows.size() && "Row and column index sizes must match");

	m_solver.n = m_size;
	m_solver.nz = internal::convert_index<MUMPS_INT8>(m_cols.size());

	m_solver.a = m_coeffs.data();
	m_solver.irn = m_rows.data();
	m_solver.jcn = m_cols.data();

	m_solver.job = internal::mumps::job::factorisation;

	solver_api_type::mumps_c(m_solver);

	// Check the returned error
	if (m_solver.info[0] < 0)
	{
		m_info = NumericalIssue;
		m_factorizationIsOk = false;
		m_isInitialized = false;
		throw std::domain_error("Error code " + std::to_string(m_solver.info[0])
		+ " in MUMPS factorize()");
	}
	else
	{
		m_info = Success;
		m_factorizationIsOk = true;
		m_isInitialized = true;
	}
}

template <typename Base>
template <typename Rhs, typename Dest>
bool MumpsBase<Base>::_solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& x) const
{
	eigen_assert(m_isInitialized && "Call factorize() first");

	EIGEN_STATIC_ASSERT((Dest::Flags & RowMajorBit) == 0,
			THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);

	// on return, x is overwritten by the computed solution
	x = b;

	m_solver.n = m_size;
	m_solver.nz = internal::convert_index<MUMPS_INT8>(m_cols.size());

	m_solver.a = m_coeffs.data();
	m_solver.irn = m_rows.data();
	m_solver.jcn = m_cols.data();

	m_solver.job = internal::mumps::job::back_substitution;

	m_solver.nrhs = 1;
	m_solver.lrhs = m_size;

	for (Index i = 0; i < b.cols(); i++)
	{
		m_solver.rhs = &x(0, i);
		solver_api_type::mumps_c(m_solver);
	}

	// Check the returned error
	m_info = m_solver.info[0] < 0 ? NumericalIssue : Success;

	if (m_info == NumericalIssue)
	{
		throw std::domain_error("Error code " + std::to_string(m_solver.info[0]) + " in MUMPS solve");
	}
	return m_solver.info[0] >= 0;
}

template <typename _MatrixType>
class MUMPSLU : public MumpsBase<MUMPSLU<_MatrixType>>
{
public:
	typedef _MatrixType MatrixType;
	typedef MumpsBase<MUMPSLU<MatrixType>> Base;
	typedef typename Base::ColSpMatrix ColSpMatrix;
	typedef typename MatrixType::StorageIndex StorageIndex;

public:
	MUMPSLU() : Base() {}

	explicit MUMPSLU(const MatrixType& matrix) : Base() { compute(matrix); }

	void compute(const MatrixType& matrix)
	{
		allocate_coordinate_format(matrix);
		Base::compute(matrix);
	}

	void analyzePattern(const MatrixType& matrix)
	{
		allocate_coordinate_format(matrix);
		Base::analyzePattern();
	}

	void factorize(const MatrixType& matrix)
	{
		allocate_coordinate_format(matrix);
		Base::factorize();
	}

protected:
	void allocate_coordinate_format(const MatrixType& matrix)
	{
		m_size = matrix.rows();

		m_rows.resize(matrix.nonZeros());
		m_cols.resize(matrix.nonZeros());

		m_coeffs = matrix.coeffs();

		// Decompress the upper part of the sparse matrix and convert to one
		// based indexing for MUMPS solver
		for (Index k = 0, l = 0; k < matrix.outerSize(); ++k)
		{
			for (typename MatrixType::InnerIterator it(matrix, k); it; ++it, ++l)
			{
				m_rows(l) = it.row();
				m_cols(l) = it.col();
			}
		}
		// Convert to one based indexing
		m_rows += 1;
		m_cols += 1;
	}
	using Base::m_coeffs;
	using Base::m_cols;
	using Base::m_rows;

	using Base::m_size;
};

template <typename _MatrixType, int _UpLo>
class MUMPSLDLT : public MumpsBase<MUMPSLDLT<_MatrixType, _UpLo>>
{
public:
	typedef _MatrixType MatrixType;
	typedef MumpsBase<MUMPSLDLT<MatrixType, _UpLo>> Base;
	typedef typename Base::ColSpMatrix ColSpMatrix;

public:
	enum { UpLo = _UpLo };
	MUMPSLDLT() : Base() { init(true); }

	explicit MUMPSLDLT(const MatrixType& matrix) : Base()
	{
		init(true);
		compute(matrix);
	}

	void compute(const MatrixType& matrix)
	{
		allocate_coordinate_format(matrix);
		Base::compute();
	}

	void analyzePattern(const MatrixType& matrix)
	{
		allocate_coordinate_format(matrix);
		Base::analyzePattern();
	}

	void factorize(const MatrixType& matrix)
	{
		allocate_coordinate_format(matrix);
		Base::factorize();
	}

protected:
	void allocate_coordinate_format(const MatrixType& matrix)
	{
		eigen_assert(matrix.rows() == matrix.cols() && "Input matrix must be square");

		m_size = matrix.rows();

		Index sym_nonzeros = 0;
		for (Index k = 0; k < matrix.outerSize(); ++k)
		{
			for (typename MatrixType::InnerIterator it(matrix, k); it; ++it)
			{
				if (it.col() >= it.row()) ++sym_nonzeros;
			}
		}

		m_rows.resize(sym_nonzeros);
		m_cols.resize(sym_nonzeros);
		m_coeffs.resize(sym_nonzeros);

		// Decompress the upper part of the sparse matrix and convert to one
		// based indexing for MUMPS solver
		for (Index k = 0, l = 0; k < matrix.outerSize(); ++k)
		{
			for (typename MatrixType::InnerIterator it(matrix, k); it; ++it)
			{
				if (it.col() >= it.row())
				{
					m_rows(l) = it.row();
					m_cols(l) = it.col();
					m_coeffs(l) = it.value();
					++l;
				}
			}
		}
		m_rows += 1;
		m_cols += 1;
	}

protected:
	using Base::init;

	using Base::m_coeffs;
	using Base::m_cols;
	using Base::m_rows;

	using Base::m_solver;

	using Base::m_size;
};
}

#endif
