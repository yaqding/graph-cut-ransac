// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Yaqing Ding (yaqing.ding@cvut.cz)
#pragma once

#include "estimators/solver_engine.h"
#include "estimators/homography_estimator.h"
#include "estimators/solver_homography_one_sift.h"
#include "math_utils.h"
#include <iostream>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class HomographyTwoScaOriSolver : public SolverEngine
			{

			public:
				HomographyTwoScaOriSolver()
				{
				}

				~HomographyTwoScaOriSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// when function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 2;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
				}

				static constexpr bool needFundamentalMatrix()
				{
					return false;
				}

				void setFundamentalMatrix(Eigen::Matrix3d &kFundamentalMatrix_)
				{
				}
				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat &data_,					 // The set of data points
					const size_t *sample_,					 // The sample used for the estimation
					size_t sampleNumber_,					 // The size of the sample
					std::vector<Model> &models_,			 // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool HomographyTwoScaOriSolver::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sampleNumber_ < sampleSize())
				{
					fprintf(stderr, "There were not enough affine correspondences provided for the solver (%d < 2).\n", sampleNumber_);
					return false;
				}

				const size_t kColumns = data_.cols;
				const double *kDataPtr = reinterpret_cast<double *>(data_.data);
				size_t rowIdx = 0;
				double weight = 1.0;
				int data_idx = 0;

				size_t i = 0;
				const size_t kIdx1 =
					sample_ == nullptr ? i : sample_[i];

				const double *kPointPtr1 =
					kDataPtr + kIdx1 * kColumns;

				const double
					&x11 = kPointPtr1[0],
					&y11 = kPointPtr1[1],
					&x12 = kPointPtr1[2],
					&y12 = kPointPtr1[3],
					&q11 = kPointPtr1[4],
					&q12 = kPointPtr1[5],
					&o11 = kPointPtr1[6],
					&o12 = kPointPtr1[7];

				const double k11 = tan(o11),
							 k12 = tan(o12);

				const double b11 = y11 - k11 * x11,
							 b12 = y12 - k12 * x12;

				i = 1;
				const size_t kIdx2 =
					sample_ == nullptr ? i : sample_[i];

				const double *kPointPtr2 =
					kDataPtr + kIdx2 * kColumns;

				const double
					&x21 = kPointPtr2[0],
					&y21 = kPointPtr2[1],
					&x22 = kPointPtr2[2],
					&y22 = kPointPtr2[3],
					&q21 = kPointPtr2[4],
					&q22 = kPointPtr2[5],
					&o21 = kPointPtr2[6],
					&o22 = kPointPtr2[7];

				const double k21 = tan(o21),
							 k22 = tan(o22);

				const double b21 = y11 - k11 * x11,
							 b22 = y12 - k12 * x12;

				const double x31 = b11 - b21, x32 = b11 * k21 - b21 * k11, x33 = k21 - k11;
				const double y31 = b12 - b22, y32 = b12 * k22 - b22 * k12, y33 = k22 - k12;
				const double s1 = q11 / q12, s2 = q21 / q22;

				Eigen::Matrix3d Z0;
				Z0 << x11, x21, x31,
					y11, y21, x32,
					1, 1, x33;
				Eigen::Matrix3d Z2 = Z0.inverse();

				Eigen::Matrix3d Z1;
				Z1 << s1 * x12, s2 * x22, y31,
					s1 * y12, s2 * y22, y32,
					s1, s2, y33;

				const double a1 = Z1(0, 0),
							 a2 = Z1(0, 1),
							 a3 = Z1(0, 2),
							 a4 = Z1(1, 0),
							 a5 = Z1(1, 1),
							 a6 = Z1(1, 2),
							 a7 = Z1(2, 0),
							 a8 = Z1(2, 1),
							 a9 = Z1(2, 2);
				const double b1 = Z2(0, 0),
							 b2 = Z2(0, 1),
							 b3 = Z2(0, 2),
							 b4 = Z2(1, 0),
							 b5 = Z2(1, 1),
							 b6 = Z2(1, 2),
							 b7 = Z2(2, 0),
							 b8 = Z2(2, 1),
							 b9 = Z2(2, 2);

				const double m0 = a1 * b1 + a2 * b4,
							 m1 = a3 * b7,
							 m2 = a1 * b2 + a2 * b5,
							 m3 = a3 * b8,
							 m4 = a1 * b3 + a2 * b6,
							 m5 = a3 * b9,
							 m6 = a4 * b1 + a5 * b4,
							 m7 = a6 * b7,
							 m8 = a4 * b2 + a5 * b5,
							 m9 = a6 * b8,
							 m10 = a4 * b3 + a5 * b6,
							 m11 = a6 * b9,
							 m12 = a7 * b1 + a8 * b4,
							 m13 = a9 * b7,
							 m14 = a7 * b2 + a8 * b5,
							 m15 = a9 * b8,
							 m16 = a7 * b3 + a8 * b6,
							 m17 = a9 * b9;

				const double d1 = a1 * a5 * a9 - a1 * a6 * a8 - a2 * a4 * a9 + a2 * a6 * a7 + a3 * a4 * a8 - a3 * a5 * a7,
							 d2 = b1 * b5 * b9 - b1 * b6 * b8 - b2 * b4 * b9 + b2 * b6 * b7 + b3 * b4 * b8 - b3 * b5 * b7,
							 c0 = m0 * m0 + m6 * m6 + m12 * m12,
							 c1 = 2.0 * m0 * m1 + 2.0 * m6 * m7 + 2.0 * m12 * m13,
							 c2 = m1 * m1 + m7 * m7 + m13 * m13,
							 c3 = m0 * m2 + m6 * m8 + m12 * m14,
							 c4 = m0 * m3 + m1 * m2 + m6 * m9 + m7 * m8 + m12 * m15 + m13 * m14,
							 c5 = m1 * m3 + m7 * m9 + m13 * m15,
							 c6 = m0 * m4 + m6 * m10 + m12 * m16,
							 c7 = m0 * m5 + m1 * m4 + m6 * m11 + m7 * m10 + m12 * m17 + m13 * m16,
							 c8 = m1 * m5 + m7 * m11 + m13 * m17,
							 c9 = m2 * m2 + m8 * m8 + m14 * m14,
							 c10 = 2.0 * m2 * m3 + 2.0 * m8 * m9 + 2.0 * m14 * m15,
							 c11 = m3 * m3 + m9 * m9 + m15 * m15,
							 c12 = m2 * m4 + m8 * m10 + m14 * m16,
							 c13 = m2 * m5 + m3 * m4 + m8 * m11 + m9 * m10 + m14 * m17 + m15 * m16,
							 c14 = m3 * m5 + m9 * m11 + m15 * m17,
							 c15 = m4 * m4 + m10 * m10 + m16 * m16,
							 c16 = 2.0 * m4 * m5 + 2.0 * m10 * m11 + 2.0 * m16 * m17,
							 c17 = m5 * m5 + m11 * m11 + m17 * m17;

				const double coe0 = -(-c3 * c3 + c0 * c9 - c6 * c6 + c0 * c15 - c12 * c12 + c9 * c15) + c0 + c9 + c15 - 1.0,
							 coe1 = -(c0 * c10 - 2.0 * c3 * c4 + c1 * c9 + c0 * c16 - 2.0 * c6 * c7 + c1 * c15 + c9 * c16 + c10 * c15 - 2.0 * c12 * c13) + c1 + c10 + c16,
							 coe2 = -(-c4 * c4 - 2.0 * c3 * c5 + c0 * c11 + c1 * c10 + c2 * c9 - c7 * c7 - 2.0 * c6 * c8 + c0 * c17 + c1 * c16 + c2 * c15 - c13 * c13 + c9 * c17 + c10 * c16 + c11 * c15 - 2.0 * c12 * c14) + c2 + c11 + c17 + d1 * d2 * d1 * d2;
				
				double roots[2];
				bool flag = gcransac::estimator::solver::solve_quadratic(coe2, coe1, coe0, roots);

				if (!flag)
				{
					return false;
				}

				Model model;
				model.descriptor.resize(3, 3);

				for (int i = 0; i < 2; ++i)
				{							
					if (roots[i] > 0)
					{
						Z1 << s1 * x12, s2 * x22, roots[i]*y31,
						s1 * y12, s2 * y22, roots[i]*y32,
						s1, s2, roots[i]*y33;
						model.descriptor = Z1 * Z2;

						models_.push_back(model);
					}
				}
				
				return true;
			}
		}
	}
}