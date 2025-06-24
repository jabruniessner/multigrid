/**
 *  @ingroup PMGC
 *  @author  Tucker Beck [fortran ->c translation], Michael Holst [original]
 *  @brief   A collection of useful low-level routines (timing, etc).
 *  @version $Id:
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

#ifndef MIKPCKD_H_
#define MIKPCKD_H_

#include "abps_macros.h"
#include <sycl/sycl.hpp>

/** @brief   Copy operation for a grid function with boundary values.
 *           Quite simply copies one 3d matrix to another
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces xcopy from mikpckd.f
 */

template <typename DataType>
void Vxcopy(int *nx,     ///< The size of the x dimension of the 3d matrix
            int *ny,     ///< The size of the y dimension of the 3d matrix
            int *nz,     ///< The size of the z dimension of the 3d matrix
            DataType *x, ///< The source matrix from which to copy data
            DataType *y,
            sycl::queue &q ///< The destination matrix to receive copied data
);

/** @brief   Copy operation for a grid function with boundary values.
 *           Quite simply copies one 3d matrix to another
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces xcopy_small from mikpckd.f
 */

template <typename DataType>
void Vxcopy_small(
    int *nx,     ///< The size of the x dimension of the 3d matrix
    int *ny,     ///< The size of the y dimension of the 3d matrix
    int *nz,     ///< The size of the z dimension of the 3d matrix
    DataType *x, ///< The source matrix from which to copy data
    DataType *y, ///< The destination matrix to receive copied data
    sycl::queue &q);

/** @brief   Copy operation for a grid function with boundary values.
 *           Quite simply copies one 3d matrix to another
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces xcopy_large from mikpckd.f
 */
template <typename DataType>
void Vxcopy_large(
    int *nx,     ///< The size of the x dimension of the 3d matrix
    int *ny,     ///< The size of the y dimension of the 3d matrix
    int *nz,     ///< The size of the z dimension of the 3d matrix
    DataType *x, ///< The source matrix from which to copy data
    DataType *y,
    sycl::queue &q ///< The destination matrix to receive copied data
);

/** @brief   saxpy operation for a grid function with boundary values.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces xaxpy from mikpckd.f
 */

template <typename DataType>
void Vxaxpy(int *nx,         ///< The size of the x dimension of the 3d matrix
            int *ny,         ///< The size of the y dimension of the 3d matrix
            int *nz,         ///< The size of the z dimension of the 3d matrix
            DataType *alpha, ///< @todo: Doc
            DataType *x,     ///< The source matrix from which to copy data
            DataType *y,
            sycl::queue &q ///< The destination matrix to receive copied data
);

/** @brief   Norm operation for a grid function with boundary values.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces xnrm1 from mikpckd.f
 */
template <typename DataType>
DataType Vxnrm1(int *nx, ///< The size of the x dimension of the 3d matrix
                int *ny, ///< The size of the y dimension of the 3d matrix
                int *nz, ///< The size of the z dimension of the 3d matrix
                DataType *x,
                sycl::queue &q ///< The matrix to normalize
);

/** @brief   Norm operation for a grid function with boundary values.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces xnrm2 from mikpckd.f
 */

template <typename DataType>
DataType Vxnrm2(int *nx, ///< The size of the x dimension of the 3d matrix
                int *ny, ///< The size of the y dimension of the 3d matrix
                int *nz, ///< The size of the z dimension of the 3d matrix
                DataType *x,
                sycl::queue &q ///< The matrix to normalize
);

/** @brief   Inner product operation for a grid function with boundary values.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces xdot from mikpckd.f
 */

template <typename DataType>
DataType Vxdot(int *nx,     ///< The size of the x dimension of the 3d matrix
               int *ny,     ///< The size of the y dimension of the 3d matrix
               int *nz,     ///< The size of the z dimension of the 3d matrix
               DataType *x, ///< The first vector
               DataType *y,
               sycl::queue &q ///< The second vector
);

/** @brief   Zero out operation for a grid function, including boundary values.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces azeros from mikpckd.f
 */

template <typename DataType>
void Vazeros(int *nx, ///< The size of the x dimension of the 3d matrix
             int *ny, ///< The size of the x dimension of the 3d matrix
             int *nz, ///< The size of the x dimension of the 3d matrix
             DataType *x,
             sycl::queue &q ///< The matrix to zero out
);

/** @brief   Initialize a grid function to have a certain boundary value,
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces fboundPMG from mikpckd.f
 */

template <typename DataType>
void VfboundPMG(int *ibound,   ///< @todo: Doc
                int *nx,       ///< @todo: Doc
                int *ny,       ///< @todo: Doc
                int *nz,       ///< @todo: Doc
                DataType *x,   ///< @todo: Doc
                DataType *gxc, ///< @todo: Doc
                DataType *gyc, ///< @todo: Doc
                DataType *gzc,
                sycl::queue &q ///< @todo: Doc
);

/** @brief   Initialize a grid function to have a zero boundary value
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces fboundPMG00 from mikpckd.f
 */

template <typename DataType>
void VfboundPMG00(int *nx, ///< The size of the x dimension of the 3d matrix
                  int *ny, ///< The size of the y dimension of the 3d matrix
                  int *nz, ///< The size of the z dimension of the 3d matrix
                  DataType *x,
                  sycl::queue &q ///< The 3d matrix to initialize
);

/** @brief   Fill grid function with random values, including boundary values.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces axrand from mikpckd.f
 */

template <typename DataType>
void Vaxrand(int *nx,    ///< The size of the x dimension of the 3d matrix
             int *ny,    ///< The size of the y dimension of the 3d matrix
             int *nz,    ///< The size of the z dimension of the 3d matrix
             DataType *x ///< The 3d matrix to fill
);

/** @brief   Scale operation for a grid function with boundary values.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces xscal from mikpckd.f
 */

template <typename DataType>
void Vxscal(int *nx,       ///< The size of the x dimension of the 3d matrix
            int *ny,       ///< The size of the y dimension of the 3d matrix
            int *nz,       ///< The size of the z dimension of the 3d matrix
            DataType *fac, ///< The scaling factor
            DataType *x,
            sycl::queue &q ///< The 3d matrix to scale
);

/** @brief
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces prtmatd from mikpckd.f
 */
template <typename DataType>
void Vprtmatd(int *nx,       ///< The size of the x dimension of the 3d matrix
              int *ny,       ///< The size of the y dimension of the 3d matrix
              int *nz,       ///< The size of the z dimension of the 3d matrix
              int *ipc,      ///< Integer parameters
              DataType *rpc, ///< Double parameters
              DataType *ac   ///< @todo  Document
);

template <typename DataType>
void Vprtmatd7(int *nx,       ///< The size of the x dimension of the 3d matrix
               int *ny,       ///< The size of the y dimension of the 3d matrix
               int *nz,       ///< The size of the z dimension of the 3d matrix
               int *ipc,      ///< Integer parameters
               DataType *rpc, ///< Double parameters
               DataType *oC,  ///< @todo  Document
               DataType *oE,  ///< @todo  Document
               DataType *oN,  ///< @todo  Document
               DataType *uC   ///< @todo  Document
);

template <typename DataType>
void Vprtmatd27(int *nx,       ///< The size of the x dimension of the 3d matrix
                int *ny,       ///< The size of the y dimension of the 3d matrix
                int *nz,       ///< The size of the z dimension of the 3d matrix
                int *ipc,      ///< Integer parameters
                DataType *rpc, ///< Double parameters
                DataType *oC,  ///< @todo  Document
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
                DataType *uSW  ///< @todo  Document
);

template <typename DataType>
void Vlinesearch(int *nx, ///< The size of the x dimension of the 3d matrix
                 int *ny, ///< The size of the y dimension of the 3d matrix
                 int *nz, ///< The size of the z dimension of the 3d matrix
                 DataType *alpha, ///< @todo  Document
                 int *ipc,        ///< Integer parameters
                 DataType *rpc,   ///< Double parameters
                 DataType *ac,    ///< @todo  Document
                 DataType *cc,    ///< @todo  Document
                 DataType *fc,    ///< @todo  Document
                 DataType *p,     ///< @todo  Document
                 DataType *x,     ///< @todo  Document
                 DataType *r,     ///< @todo  Document
                 DataType *ap,    ///< @todo  Document
                 DataType *zk,    ///< @todo  Document
                 DataType *zkp1   ///< @todo  Document
);

#endif /* MIKPCKD_H_ */
