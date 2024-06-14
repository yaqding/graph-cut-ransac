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
#include "math_utils.h"
#include <iostream>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class HomographyOneSIFTSolver : public SolverEngine
			{
			protected:
			public:
				HomographyOneSIFTSolver() 
				{
				}

				~HomographyOneSIFTSolver()
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
					return 1;
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

			OLGA_INLINE bool solve_quadratic(double a, double b, double c, double roots[2]) {

				double b24ac = b * b - 4.0 * a * c;
				if (b24ac < 0)
					return false;

				double sq = std::sqrt(b24ac);

				roots[0] = (b > 0) ? (2.0 * c) / (-b - sq) : (2.0 * c) / (-b + sq);
				roots[1] = c / (a * roots[0]);
				return true;
				}

			OLGA_INLINE bool HomographyOneSIFTSolver::estimateModel(
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

				// The number of equations per correspondences
				// The row number of the coefficient matrix

				const size_t kColumns = data_.cols;
				const double *kDataPtr = reinterpret_cast<double *>(data_.data);
				size_t rowIdx = 0;
				double weight = 1.0;
				int data_idx = 0;

				size_t i = 0;
				const size_t kIdx =
					sample_ == nullptr ? i : sample_[i];

				const double *kPointPtr =
					kDataPtr + kIdx * kColumns;

				const double
					&u1 = kPointPtr[0],
					&v1 = kPointPtr[1],
					&u2 = kPointPtr[2],
					&v2 = kPointPtr[3],
					&q01 = kPointPtr[4],
					&q02 = kPointPtr[5],
					&o1 = kPointPtr[6],
					&o2 = kPointPtr[7];

				if (weights_ != nullptr)
					weight = weights_[kIdx];

				const double a1 = q02 / q01 * cos(o2-o1);
				const double a2 = q02 / q01 * sin(o1-o2);

				const double k1 = tan(o1);

				const double s = q01 / q02;

				const double p0 = a1*s;
				const double p1 = a2*s;

				const double p4 = -k1;
				const double p5 = -(p0 / u1 - cos(o2) / (cos(o1)*u1) + p1*k1 / u1);
				const double p6 = -(p4*u1*u1 + u1*v1);
				const double p7 = -(p0*u1 + p1*v1 - s*u2 + p5*u1*u1);
				const double p8 = -(v1*v1 + p4*u1*v1);
				const double p9 = -(-p1*u1 + p0*v1 - s*v2+ p5*u1*v1);

				const double q0 = -(v1 + p4*u1);
				const double q1 = s-p5*u1;
				const double q2 = p4*u1; 
				const double q3 = p0 + p5*u1;
				const double q4 = p4*v1; 
				const double q5 = -p1 + p5*v1;

				const double t0 = p4*p4 + q2*q2 + q4*q4;
				const double r0 = 2*p4*p5 + 2*q2*q3 + 2*q4*q5;
				const double s0 = p5*p5 + q3*q3 + q5*q5 - 1.0;

				const double t1 = p4 + q2*u1 + q4*v1; 
				const double r1 = p5 + p1*q2 + p0*q4 + q3*u1 + q5*v1;
				const double s1 = p1*q3 + p0*q5;

				const double t2 = p4*q0 + p6*q2 + p8*q4;
				const double r2 = p4*q1 + p5*q0 + p6*q3 + p7*q2 + p8*q5 + p9*q4;
				const double s2 = p5*q1 + p7*q3 + p9*q5;

				const double t4 = u1*u1 + v1*v1 + 1.0;
				const double r4 = 2*p1*u1 + 2*p0*v1; 
				const double s4 = p1*p1 + p0*p0 - 1;

				const double t5 = q0 + p6*u1 + p8*v1; 
				const double r5 = q1 + p1*p6 + p0*p8 + p7*u1 + p9*v1;
				const double s5 = p1*p7 + p0*p9;

				const double t8 = p6*p6 + p8*p8 + q0*q0;
				const double r8 = 2*p6*p7 + 2*p8*p9 + 2*q0*q1;
				const double s8 = p7*p7 + p9*p9 + q1*q1 - 1;

				const double coe0 = - s8*s1*s1 + 2*s1*s2*s5 - s4*s2*s2 - s0*s5*s5 + s0*s4*s8;
				const double coe1 = - r8*s1*s1 + 2*r5*s1*s2 + 2*r2*s1*s5 - 2*r1*s8*s1 - r4*s2*s2 + 2*r1*s2*s5 - 2*r2*s4*s2 - r0*s5*s5 - 2*r5*s0*s5 + r0*s4*s8 + r4*s0*s8 + r8*s0*s4;
				const double coe2 = - s8*r1*r1 + 2*r1*r2*s5 + 2*r1*r5*s2 - 2*r8*r1*s1 - s4*r2*r2 + 2*r2*r5*s1 - 2*r4*r2*s2 - s0*r5*r5 - 2*r0*r5*s5 - t8*s1*s1 + 2*t5*s1*s2 + 2*t2*s1*s5 - 2*s8*t1*s1 - t4*s2*s2 + 2*t1*s2*s5 - 2*s4*t2*s2 - t0*s5*s5 - 2*s0*t5*s5 + r0*r4*s8 + r0*r8*s4 + r4*r8*s0 + s0*s4*t8 + s0*s8*t4 + s4*s8*t0;

				double roots[2];
				const bool flag = solve_quadratic(coe2, coe1, coe0, roots);

				if (!flag)
				{
					return false;
				}

				Model model;
				model.descriptor.resize(3, 3);

				for (int i = 0; i < 2; ++i)
				{
					model.descriptor(0, 0) = q2*roots[i]+q3;
					model.descriptor(0, 1) = u1*roots[i]+p1;
					model.descriptor(0, 2) = p6*roots[i]+p7;
					model.descriptor(1, 0) = q4*roots[i]+q5;
					model.descriptor(1, 1) = v1*roots[i]+p0;
					model.descriptor(1, 2) = p8*roots[i]+p9;
					model.descriptor(2, 0) = p4*roots[i]+p5;
					model.descriptor(2, 1) = roots[i];
					model.descriptor(2, 2) = q0*roots[i]+q1;

					models_.push_back(model);
				}

				return true;
			}
		}
	}
}