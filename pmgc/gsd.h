/**
 *  @ingroup PMGC
 *  @author  Tucker Beck [fortran ->c translation], Michael Holst [original]
 *  @brief Guass-Seidel solver
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

#ifndef _GSD_H_
#define _GSD_H_

#include "apbs_macros.h"
// #include "matvecd.h"
#include <sycl/sycl.hpp>

/** @brief   Call the fast diagonal iterative method.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces gsrb from gsd.f
 */

namespace pmgc {

template <typename DataType>
void Vgsrb(int *nx,       ///< @todo:  Doc
           int *ny,       ///< @todo:  Doc
           int *nz,       ///< @todo:  Doc
           int *ipc,      ///< @todo:  Doc
           DataType *rpc, ///< @todo:  Doc
           DataType *ac,  ///< @todo:  Doc
           DataType *cc,  ///< @todo:  Doc
           DataType *fc,  ///< @todo:  Doc
           DataType *x,   ///< @todo:  Doc
           int *itmax,    ///< @todo:  Doc
           sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vgsrb7x(int *nx,       ///< @todo:  Doc
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
             int *itmax,    ///< @todo:  Doc
             sycl::queue &q ///< @todo:  Doc
);

template <typename DataType>
void Vgsrb27x(int *nx,       ///< @todo:  Doc
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
              int *itmax,    ///< @todo:  Doc
              sycl::queue &q ///< @todo:  Doc
);

} // namespace pmgc

#endif /* _GSD_H_ */
