/**
 *  @ingroup PMGC
 *  @author  Tucker Beck [fortran ->c translation], Michael Holst [original]
 *  @brief  Matrix-vector multiplication routines
 *
 *  @attention
 *  @verbatim
 *
 * APBS -- Adaptive Poisson-Boltzmann Solver
 *
 * Nathan A. Baker (nathan.baker@pnl.gov)
 * Pacific Northwest National Laboratory
 *
 * Additional contributing authors listed in the code documentation.
 *
 * Copyright (c) 2010-2020 Battelle Memorial Institute. Developed at the Pacific
 * Northwest National Laboratory, operated by Battelle Memorial Institute,
 * Pacific Northwest Division for the U.S. Department Energy.  Portions
 * Copyright (c) 2002-2010, Washington University in St. Louis.  Portions
 * Copyright (c) 2002-2010, Nathan A. Baker.  Portions Copyright (c) 1999-2002,
 * The Regents of the University of California. Portions Copyright (c) 1995,
 * Michael Holst. All rights reserved.
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * -  Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * - Neither the name of Washington University in St. Louis nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @endverbatim
 */

#ifndef _MATVECD_H_
#define _MATVECD_H_

#include "apbs_macros.h"

#include "mikpckd.h"
#include "mypdec.h"
#include "precision.h"

#include <sycl/sycl.hpp>

/** @brief   Break the matrix data-structure into diagonals and
 *           then call the matrix-vector routine.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces matvec from matvecd.f
 */

namespace pmgc {

template <typename DataType>
void Vmatvec(int *nx,       ///< @todo:  Doc
             int *ny,       ///< @todo:  Doc
             int *nz,       ///< @todo:  Doc
             int *ipc,      ///< @todo:  Doc
             DataType *rpc, ///< @todo:  Doc
             DataType *ac,  ///< @todo:  Doc
             DataType *cc,  ///< @todo:  Doc
             DataType *x,   ///< @todo:  Doc
             DataType *y,
             sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vmatvec7(int *nx,       ///< @todo:  Doc
              int *ny,       ///< @todo:  Doc
              int *nz,       ///< @todo:  Doc
              int *ipc,      ///< @todo:  Doc
              DataType *rpc, ///< @todo:  Doc
              DataType *ac,  ///< @todo:  Doc
              DataType *cc,  ///< @todo:  Doc
              DataType *x,   ///< @todo:  Doc
              DataType *y,
              sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vmatvec7_1s(int *nx,       ///< @todo:  Doc
                 int *ny,       ///< @todo:  Doc
                 int *nz,       ///< @todo:  Doc
                 int *ipc,      ///< @todo:  Doc
                 DataType *rpc, ///< @todo:  Doc
                 DataType *oC,  ///< @todo:  Doc
                 DataType *cc,  ///< @todo:  Doc
                 DataType *oE,  ///< @todo:  Doc
                 DataType *oN,  ///< @todo:  Doc
                 DataType *uC,  ///< @todo:  Doc
                 DataType *x,   ///< @todo:  Doc
                 DataType *y,
                 sycl::queue &q ///< @todo:  Doc
);

template <std::size_t nx, std::size_t ny, std::size_t nz, typename DataType>
DataType matveckernel7(int i, int j, int k, DataType *oC, DataType *cc,
                       DataType *oE, DataType *oN, DataType *uC, DataType *x) {

  MAT3(oE, nx, ny, nz);
  MAT3(oN, nx, ny, nz);
  MAT3(uC, nx, ny, nz);
  MAT3(cc, nx, ny, nz);
  MAT3(oC, nx, ny, nz);
  MAT3(x, nx, ny, nz);
  MAT3(y, nx, ny, nz);

  // Do it

  return -VAT3(oN, i, j, k) * VAT3(x, i, j + 1, k) -
         VAT3(oN, i, j - 1, k) * VAT3(x, i, j - 1, k) -
         VAT3(oE, i, j, k) * VAT3(x, i + 1, j, k) -
         VAT3(oE, i - 1, j, k) * VAT3(x, i - 1, j, k) -
         VAT3(uC, i, j, k - 1) * VAT3(x, i, j, k - 1) -
         VAT3(uC, i, j, k) * VAT3(x, i, j, k + 1) +
         (VAT3(oC, i, j, k) + VAT3(cc, i, j, k)) * VAT3(x, i, j, k);
}

template <typename DataType>
void Vmatvec27(int *nx,       ///< @todo:  Doc
               int *ny,       ///< @todo:  Doc
               int *nz,       ///< @todo:  Doc
               int *ipc,      ///< @todo:  Doc
               DataType *rpc, ///< @todo:  Doc
               DataType *ac,  ///< @todo:  Doc
               DataType *cc,  ///< @todo:  Doc
               DataType *x,   ///< @todo:  Doc
               DataType *y,
               sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vmatvec27_1s(int *nx,       ///< @todo:  Doc
                  int *ny,       ///< @todo:  Doc
                  int *nz,       ///< @todo:  Doc
                  int *ipc,      ///< @todo:  Doc
                  DataType *rpc, ///< @todo:  Doc
                  DataType *oC,  ///< @todo:  Doc
                  DataType *cc,  ///< @todo:  Doc
                  DataType *oE,  ///< @todo:  Doc
                  DataType *oN,  ///< @todo:  Doc
                  DataType *uC,  ///< @todo:  Doc
                  DataType *oNE, ///< @todo:  Doc
                  DataType *oNW, ///< @todo:  Doc
                  DataType *uE,  ///< @todo:  Doc
                  DataType *uW,  ///< @todo:  Doc
                  DataType *uN,  ///< @todo:  Doc
                  DataType *uS,  ///< @todo:  Doc
                  DataType *uNE, ///< @todo:  Doc
                  DataType *uNW, ///< @todo:  Doc
                  DataType *uSE, ///< @todo:  Doc
                  DataType *uSW, ///< @todo:  Doc
                  DataType *x,   ///< @todo:  Doc
                  DataType *y,
                  sycl::queue &q ///< @todo:  Doc
);

template <std::size_t nx, std::size_t ny, std::size_t nz, typename DataType>
DataType matveckernel27(int i, int j, int k, DataType *oC, DataType *cc,
                        DataType *oE, DataType *oN, DataType *uC, DataType *oNE,
                        DataType *oNW, DataType *uE, DataType *uW, DataType *uN,
                        DataType *uS, DataType *uNE, DataType *uNW,
                        DataType *uSE, DataType *uSW, DataType *x) {

  MAT3(cc, nx, ny, nz);
  MAT3(x, nx, ny, nz);
  MAT3(y, nx, ny, nz);

  MAT3(oC, nx, ny, nz);
  MAT3(oE, nx, ny, nz);
  MAT3(oN, nx, ny, nz);
  MAT3(oNE, nx, ny, nz);
  MAT3(oNW, nx, ny, nz);

  MAT3(uC, nx, ny, nz);
  MAT3(uE, nx, ny, nz);
  MAT3(uW, nx, ny, nz);
  MAT3(uN, nx, ny, nz);
  MAT3(uS, nx, ny, nz);
  MAT3(uNE, nx, ny, nz);
  MAT3(uNW, nx, ny, nz);
  MAT3(uSE, nx, ny, nz);
  MAT3(uSW, nx, ny, nz);

  // Do it

  const auto tmpO = -VAT3(oN, i, j, k) * VAT3(x, i, j + 1, k) -
                    VAT3(oN, i, j - 1, k) * VAT3(x, i, j - 1, k) -
                    VAT3(oE, i, j, k) * VAT3(x, i + 1, j, k) -
                    VAT3(oE, i - 1, j, k) * VAT3(x, i - 1, j, k) -
                    VAT3(oNE, i, j, k) * VAT3(x, i + 1, j + 1, k) -
                    VAT3(oNW, i, j, k) * VAT3(x, i - 1, j + 1, k) -
                    VAT3(oNW, i + 1, j - 1, k) * VAT3(x, i + 1, j - 1, k) -
                    VAT3(oNE, i - 1, j - 1, k) * VAT3(x, i - 1, j - 1, k);

  const auto tmpU = -VAT3(uC, i, j, k) * VAT3(x, i, j, k + 1) -
                    VAT3(uN, i, j, k) * VAT3(x, i, j + 1, k + 1) -
                    VAT3(uS, i, j, k) * VAT3(x, i, j - 1, k + 1) -
                    VAT3(uE, i, j, k) * VAT3(x, i + 1, j, k + 1) -
                    VAT3(uW, i, j, k) * VAT3(x, i - 1, j, k + 1) -
                    VAT3(uNE, i, j, k) * VAT3(x, i + 1, j + 1, k + 1) -
                    VAT3(uNW, i, j, k) * VAT3(x, i - 1, j + 1, k + 1) -
                    VAT3(uSE, i, j, k) * VAT3(x, i + 1, j - 1, k + 1) -
                    VAT3(uSW, i, j, k) * VAT3(x, i - 1, j - 1, k + 1);

  const auto tmpD =
      -VAT3(uC, i, j, k - 1) * VAT3(x, i, j, k - 1) -
      VAT3(uS, i, j + 1, k - 1) * VAT3(x, i, j + 1, k - 1) -
      VAT3(uN, i, j - 1, k - 1) * VAT3(x, i, j - 1, k - 1) -
      VAT3(uW, i + 1, j, k - 1) * VAT3(x, i + 1, j, k - 1) -
      VAT3(uE, i - 1, j, k - 1) * VAT3(x, i - 1, j, k - 1) -
      VAT3(uSW, i + 1, j + 1, k - 1) * VAT3(x, i + 1, j + 1, k - 1) -
      VAT3(uSE, i - 1, j + 1, k - 1) * VAT3(x, i - 1, j + 1, k - 1) -
      VAT3(uNW, i + 1, j - 1, k - 1) * VAT3(x, i + 1, j - 1, k - 1) -
      VAT3(uNE, i - 1, j - 1, k - 1) * VAT3(x, i - 1, j - 1, k - 1);

  return tmpO + tmpU + tmpD +
         (VAT3(oC, i, j, k) + VAT3(cc, i, j, k)) * VAT3(x, i, j, k);
}

/** @brief   Break the matrix data-structure into diagonals and
 *           then call the matrix-vector routine.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces nmatvec from matvecd.f
 */

template <typename DataType>
void Vnmatvec(int *nx,       ///< @todo  Document
              int *ny,       ///< @todo  Document
              int *nz,       ///< @todo  Document
              int *ipc,      ///< @todo  Document
              DataType *rpc, ///< @todo  Document
              DataType *ac,  ///< @todo  Document
              DataType *cc,  ///< @todo  Document
              DataType *x,   ///< @todo  Document
              DataType *y,   ///< @todo  Document
              DataType *w1,
              sycl::queue &q ///< @todo  Document
);

template <typename DataType>
void Vnmatvec7(int *nx,       ///< @todo  Document
               int *ny,       ///< @todo  Document
               int *nz,       ///< @todo  Document
               int *ipc,      ///< @todo  Document
               DataType *rpc, ///< @todo  Document
               DataType *ac,  ///< @todo  Document
               DataType *cc,  ///< @todo  Document
               DataType *x,   ///< @todo  Document
               DataType *y,   ///< @todo  Document
               DataType *w1,
               sycl::queue &q ///< @todo  Document
);

template <typename DataType>
void Vnmatvecd7_1s(int *nx,       ///< @todo  Document
                   int *ny,       ///< @todo  Document
                   int *nz,       ///< @todo  Document
                   int *ipc,      ///< @todo  Document
                   DataType *rpc, ///< @todo  Document
                   DataType *oC,  ///< @todo  Document
                   DataType *cc,  ///< @todo  Document
                   DataType *oE,  ///< @todo  Document
                   DataType *oN,  ///< @todo  Document
                   DataType *uC,  ///< @todo  Document
                   DataType *x,   ///< @todo  Document
                   DataType *y,   ///< @todo  Document
                   DataType *w1,
                   sycl::queue &q ///< @todo  Document
);

template <typename DataType>
void Vnmatvec27(int *nx,       ///< @todo  Document
                int *ny,       ///< @todo  Document
                int *nz,       ///< @todo  Document
                int *ipc,      ///< @todo  Document
                DataType *rpc, ///< @todo  Document
                DataType *ac,  ///< @todo  Document
                DataType *cc,  ///< @todo  Document
                DataType *x,   ///< @todo  Document
                DataType *y,   ///< @todo  Document
                DataType *w1,
                sycl::queue &q ///< @todo  Document
);

template <typename DataType>
void Vnmatvecd27_1s(int *nx,       ///< @todo  Document
                    int *ny,       ///< @todo  Document
                    int *nz,       ///< @todo  Document
                    int *ipc,      ///< @todo  Document
                    DataType *rpc, ///< @todo  Document
                    DataType *oC,  ///< @todo  Document
                    DataType *cc,  ///< @todo  Document
                    DataType *oE,  ///< @todo  Document
                    DataType *oN,  ///< @todo  Document
                    DataType *uC,  ///< @todo  Document
                    DataType *oNE, ///< @todo  Document
                    DataType *oNW, ///< @todo  Document
                    DataType *uE,  ///< @todo  Document
                    DataType *uW,  ///< @todo  Document
                    DataType *uN,  ///< @todo  Document
                    DataType *uS,  ///< @todo  Document
                    DataType *uNE, ///< @todo  Document
                    DataType *uNW, ///< @todo  Document
                    DataType *uSE, ///< @todo  Document
                    DataType *uSW, ///< @todo  Document
                    DataType *x,   ///< @todo  Document
                    DataType *y,   ///< @todo  Document
                    DataType *w1,
                    sycl::queue &q ///< @todo  Document
);

/** @brief   Break the matrix data-structure into diagonals and
 *           then call the residual routine.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces mresid from matvecd.f
 */

template <typename DataType>
void Vmresid(int *nx,       ///< @todo:  Doc
             int *ny,       ///< @todo:  Doc
             int *nz,       ///< @todo:  Doc
             int *ipc,      ///< @todo:  Doc
             DataType *rpc, ///< @todo:  Doc
             DataType *ac,  ///< @todo:  Doc
             DataType *cc,  ///< @todo:  Doc
             DataType *fc,  ///< @todo:  Doc
             DataType *x,   ///< @todo:  Doc
             DataType *r,
             sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vmresid7(int *nx,       ///< @todo:  Doc
              int *ny,       ///< @todo:  Doc
              int *nz,       ///< @todo:  Doc
              int *ipc,      ///< @todo:  Doc
              DataType *rpc, ///< @todo:  Doc
              DataType *ac,  ///< @todo:  Doc
              DataType *cc,  ///< @todo:  Doc
              DataType *fc,  ///< @todo:  Doc
              DataType *x,   ///< @todo:  Doc
              DataType *r,
              sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vmresid7_1s(int *nx,       ///< @todo:  Doc
                 int *ny,       ///< @todo:  Doc
                 int *nz,       ///< @todo:  Doc
                 int *ipc,      ///< @todo:  Doc
                 DataType *rpc, ///< @todo:  Doc
                 DataType *oC,  ///< @todo:  Doc
                 DataType *cc,  ///< @todo:  Doc
                 DataType *fc,  ///< @todo:  Doc
                 DataType *oE,  ///< @todo:  Doc
                 DataType *oN,  ///< @todo:  Doc
                 DataType *uC,  ///< @todo:  Doc
                 DataType *x,   ///< @todo:  Doc
                 DataType *r,
                 sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vmresid27(int *nx,       ///< @todo:  Doc
               int *ny,       ///< @todo:  Doc
               int *nz,       ///< @todo:  Doc
               int *ipc,      ///< @todo:  Doc
               DataType *rpc, ///< @todo:  Doc
               DataType *ac,  ///< @todo:  Doc
               DataType *cc,  ///< @todo:  Doc
               DataType *fc,  ///< @todo:  Doc
               DataType *x,   ///< @todo:  Doc
               DataType *r,
               sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vmresid27_1s(int *nx,       ///< @todo:  Doc
                  int *ny,       ///< @todo:  Doc
                  int *nz,       ///< @todo:  Doc
                  int *ipc,      ///< @todo:  Doc
                  DataType *rpc, ///< @todo:  Doc
                  DataType *oC,  ///< @todo:  Doc
                  DataType *cc,  ///< @todo:  Doc
                  DataType *fc,  ///< @todo:  Doc
                  DataType *oE,  ///< @todo:  Doc
                  DataType *oN,  ///< @todo:  Doc
                  DataType *uC,  ///< @todo:  Doc
                  DataType *oNE, ///< @todo:  Doc
                  DataType *oNW, ///< @todo:  Doc
                  DataType *uE,  ///< @todo:  Doc
                  DataType *uW,  ///< @todo:  Doc
                  DataType *uN,  ///< @todo:  Doc
                  DataType *uS,  ///< @todo:  Doc
                  DataType *uNE, ///< @todo:  Doc
                  DataType *uNW, ///< @todo:  Doc
                  DataType *uSE, ///< @todo:  Doc
                  DataType *uSW, ///< @todo:  Doc
                  DataType *x,   ///< @todo:  Doc
                  DataType *r,
                  sycl::queue &q ///< @todo:  Doc
);

/** @brief   Break the matrix data-structure into diagonals and
 *           then call the residual routine.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces nmresid from matvecd.f
 */

template <typename DataType>
void Vnmresid(int *nx,       ///< @todo:  Doc
              int *ny,       ///< @todo:  Doc
              int *nz,       ///< @todo:  Doc
              int *ipc,      ///< @todo:  Doc
              DataType *rpc, ///< @todo:  Doc
              DataType *ac,  ///< @todo:  Doc
              DataType *cc,  ///< @todo:  Doc
              DataType *fc,  ///< @todo:  Doc
              DataType *x,   ///< @todo:  Doc
              DataType *r,   ///< @todo:  Doc
              DataType *w1,
              sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vnmresid7(int *nx,       ///< @todo:  Doc
               int *ny,       ///< @todo:  Doc
               int *nz,       ///< @todo:  Doc
               int *ipc,      ///< @todo:  Doc
               DataType *rpc, ///< @todo:  Doc
               DataType *ac,  ///< @todo:  Doc
               DataType *cc,  ///< @todo:  Doc
               DataType *fc,  ///< @todo:  Doc
               DataType *x,   ///< @todo:  Doc
               DataType *r,   ///< @todo:  Doc
               DataType *w1,
               sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vnmresid7_1s(int *nx,       ///< @todo:  Doc
                  int *ny,       ///< @todo:  Doc
                  int *nz,       ///< @todo:  Doc
                  int *ipc,      ///< @todo:  Doc
                  DataType *rpc, ///< @todo:  Doc
                  DataType *oC,  ///< @todo:  Doc
                  DataType *cc,  ///< @todo:  Doc
                  DataType *fc,  ///< @todo:  Doc
                  DataType *oE,  ///< @todo:  Doc
                  DataType *oN,  ///< @todo:  Doc
                  DataType *uC,  ///< @todo:  Doc
                  DataType *x,   ///< @todo:  Doc
                  DataType *r,   ///< @todo:  Doc
                  DataType *w1,
                  sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vnmresid27(int *nx,       ///< @todo:  Doc
                int *ny,       ///< @todo:  Doc
                int *nz,       ///< @todo:  Doc
                int *ipc,      ///< @todo:  Doc
                DataType *rpc, ///< @todo:  Doc
                DataType *ac,  ///< @todo:  Doc
                DataType *cc,  ///< @todo:  Doc
                DataType *fc,  ///< @todo:  Doc
                DataType *x,   ///< @todo:  Doc
                DataType *r,   ///< @todo:  Doc
                DataType *w1,
                sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vnmresid27_1s(int *nx,       ///< @todo:  Doc
                   int *ny,       ///< @todo:  Doc
                   int *nz,       ///< @todo:  Doc
                   int *ipc,      ///< @todo:  Doc
                   DataType *rpc, ///< @todo:  Doc
                   DataType *oC,  ///< @todo:  Doc
                   DataType *cc,  ///< @todo:  Doc
                   DataType *fc,  ///< @todo:  Doc
                   DataType *oE,  ///< @todo:  Doc
                   DataType *oN,  ///< @todo:  Doc
                   DataType *uC,  ///< @todo:  Doc
                   DataType *oNE, ///< @todo:  Doc
                   DataType *oNW, ///< @todo:  Doc
                   DataType *uE,  ///< @todo:  Doc
                   DataType *uW,  ///< @todo:  Doc
                   DataType *uN,  ///< @todo:  Doc
                   DataType *uS,  ///< @todo:  Doc
                   DataType *uNE, ///< @todo:  Doc
                   DataType *uNW, ///< @todo:  Doc
                   DataType *uSE, ///< @todo:  Doc
                   DataType *uSW, ///< @todo:  Doc
                   DataType *x,   ///< @todo:  Doc
                   DataType *r,   ///< @todo:  Doc
                   DataType *w1,
                   sycl::queue &q ///< @todo:  Doc
);

/** @brief   Apply the restriction operator
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces restrc from matvecd.f
 */

template <typename DataType>
void Vrestrc(const int nxf,  ///< @todo:  Doc
             const int nyf,  ///< @todo:  Doc
             const int nzf,  ///< @todo:  Doc
             const int nxc,  ///< @todo:  Doc
             const int nyc,  ///< @todo:  Doc
             const int nzc,  ///< @todo:  Doc
             DataType *xin,  ///< @todo:  Doc
             DataType *xout, ///< @todo:  Doc
             DataType *pc,
             sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vrestrc2(int nxf,        ///< @todo:  Doc
              int nyf,        ///< @todo:  Doc
              int nzf,        ///< @todo:  Doc
              int nxc,        ///< @todo:  Doc
              int nyc,        ///< @todo:  Doc
              int nzc,        ///< @todo:  Doc
              DataType *xin,  ///< @todo:  Doc
              DataType *xout, ///< @todo:  Doc
              DataType *oPC,  ///< @todo:  Doc
              DataType *oPN,  ///< @todo:  Doc
              DataType *oPS,  ///< @todo:  Doc
              DataType *oPE,  ///< @todo:  Doc
              DataType *oPW,  ///< @todo:  Doc
              DataType *oPNE, ///< @todo:  Doc
              DataType *oPNW, ///< @todo:  Doc
              DataType *oPSE, ///< @todo:  Doc
              DataType *oPSW, ///< @todo:  Doc
              DataType *uPC,  ///< @todo:  Doc
              DataType *uPN,  ///< @todo:  Doc
              DataType *uPS,  ///< @todo:  Doc
              DataType *uPE,  ///< @todo:  Doc
              DataType *uPW,  ///< @todo:  Doc
              DataType *uPNE, ///< @todo:  Doc
              DataType *uPNW, ///< @todo:  Doc
              DataType *uPSE, ///< @todo:  Doc
              DataType *uPSW, ///< @todo:  Doc
              DataType *dPC,  ///< @todo:  Doc
              DataType *dPN,  ///< @todo:  Doc
              DataType *dPS,  ///< @todo:  Doc
              DataType *dPE,  ///< @todo:  Doc
              DataType *dPW,  ///< @todo:  Doc
              DataType *dPNE, ///< @todo:  Doc
              DataType *dPNW, ///< @todo:  Doc
              DataType *dPSE, ///< @todo:  Doc
              DataType *dPSW,
              sycl::queue &q ///< @todo:  Doc
);

/** @brief   Apply the prolongation operator
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces interpPMG from matvecd.f
 */

template <typename DataType>
void VinterpPMG(const int nxc,  ///< @todo:  Doc
                const int nyc,  ///< @todo:  Doc
                const int nzc,  ///< @todo:  Doc
                const int nxf,  ///< @todo:  Doc
                const int nyf,  ///< @todo:  Doc
                const int nzf,  ///< @todo:  Doc
                DataType *xin,  ///< @todo:  Doc
                DataType *xout, ///< @todo:  Doc
                DataType *pc,
                sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void VinterpPMG2(const int nxc,  ///< @todo:  Doc
                 const int nyc,  ///< @todo:  Doc
                 const int nzc,  ///< @todo:  Doc
                 const int nxf,  ///< @todo:  Doc
                 const int nyf,  ///< @todo:  Doc
                 const int nzf,  ///< @todo:  Doc
                 DataType *xin,  ///< @todo:  Doc
                 DataType *xout, ///< @todo:  Doc
                 DataType *oPC,  ///< @todo:  Doc
                 DataType *oPN,  ///< @todo:  Doc
                 DataType *oPS,  ///< @todo:  Doc
                 DataType *oPE,  ///< @todo:  Doc
                 DataType *oPW,  ///< @todo:  Doc
                 DataType *oPNE, ///< @todo:  Doc
                 DataType *oPNW, ///< @todo:  Doc
                 DataType *oPSE, ///< @todo:  Doc
                 DataType *oPSW, ///< @todo:  Doc
                 DataType *uPC,  ///< @todo:  Doc
                 DataType *uPN,  ///< @todo:  Doc
                 DataType *uPS,  ///< @todo:  Doc
                 DataType *uPE,  ///< @todo:  Doc
                 DataType *uPW,  ///< @todo:  Doc
                 DataType *uPNE, ///< @todo:  Doc
                 DataType *uPNW, ///< @todo:  Doc
                 DataType *uPSE, ///< @todo:  Doc
                 DataType *uPSW, ///< @todo:  Doc
                 DataType *dPC,  ///< @todo:  Doc
                 DataType *dPN,  ///< @todo:  Doc
                 DataType *dPS,  ///< @todo:  Doc
                 DataType *dPE,  ///< @todo:  Doc
                 DataType *dPW,  ///< @todo:  Doc
                 DataType *dPNE, ///< @todo:  Doc
                 DataType *dPNW, ///< @todo:  Doc
                 DataType *dPSE, ///< @todo:  Doc
                 DataType *dPSW,
                 sycl::queue &q ///< @todo:  Doc
);

/** @brief   Simple injection of a fine grid function into coarse grid.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces extrac from matvecd.f
 */

template <typename DataType>
void Vextrac(int *nxf,      ///< @todo:  Doc
             int *nyf,      ///< @todo:  Doc
             int *nzf,      ///< @todo:  Doc
             int *nxc,      ///< @todo:  Doc
             int *ny,       ///< @todo:  Doc
             int *nzc,      ///< @todo:  Doc
             DataType *xin, ///< @todo:  Doc
             DataType *xout,
             sycl::queue &q ///< @todo:  Doc
);

} // namespace pmgc

#endif /* _MATVECD_H_ */
