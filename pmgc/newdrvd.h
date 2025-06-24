/**
 *  @ingroup PMGC
 *  @author  Tucker Beck [fortran ->c translation], Michael Holst [original]
 *  @brief   Driver for the Newton Solver
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

#ifndef _NEWDRVD_H_
#define _NEWDRVD_H_

#include "abps_macros.h"

#include "mgdrvd.h"
#include "mgsubd.h"
#include "mikpckd.h"
#include "newtond.h"

/** @brief   Driver for a screaming inexact-newton-multilevel solver.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *  @note    Replaces newdriv from newdrvd.f
 */

template <typename DataType>
void Vnewdriv(int *iparm,      ///< @todo:  Doc
              DataType *rparm, ///< @todo:  Doc
              int *iwork,      ///< @todo:  Doc
              DataType *rwork, ///< @todo:  Doc
              DataType *u,     ///< @todo:  Doc
              DataType *xf,    ///< @todo:  Doc
              DataType *yf,    ///< @todo:  Doc
              DataType *zf,    ///< @todo:  Doc
              DataType *gxcf,  ///< @todo:  Doc
              DataType *gycf,  ///< @todo:  Doc
              DataType *gzcf,  ///< @todo:  Doc
              DataType *a1cf,  ///< @todo:  Doc
              DataType *a2cf,  ///< @todo:  Doc
              DataType *a3cf,  ///< @todo:  Doc
              DataType *ccf,   ///< @todo:  Doc
              DataType *fcf,   ///< @todo:  Doc
              DataType *tcf,
              sycl::queue &q ///< @todo:  Doc
);

/** @brief   Solves using Newton's Method
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *
 *    This routine uses a newton's method, combined with a linear
 *    multigrid iteration, to solve the following three-dimensional,
 *    2nd order elliptic partial differential equation:
 *
 *         lu = f, u in omega
 *          u = g, u on boundary of omega
 *    where
 *
 *         omega = [xmin,xmax]x[ymin,ymax]x[zmin,zmax]
 *
 *    the multigrid code requires the operator in the form:
 *
 *         - \nabla \cdot (a \nabla u) + c(u) = f
 *
 *    with
 *
 *        a(x,y,z),f(x,y,z), scalar functions (possibly discontinuous)
 *        on omega.  (discontinuities must be along fine grid lines).
 *        boundary function g(x,y,z) is smooth on boundary of omega.
 *
 *        the function c(u) is a possibly nonlinear function of the
 *        unknown u, and varies (possibly discontinuously) with the
 *        spatial position also.
 *
 *   User inputs:
 *
 *    the user must provide the coefficients of the differential
 *    operator, some initial parameter settings in an integer and a
 *    real parameter array, and various work arrays.
 *
 *  @note    Replaces newdriv2 from newdrvd.f
 */
template <typename DataType>
void Vnewdriv2(int *iparm,      ///< @todo:  Doc
               DataType *rparm, ///< @todo:  Doc
               int *nx,         ///< @todo:  Doc
               int *ny,         ///< @todo:  Doc
               int *nz,         ///< @todo:  Doc
               DataType *u,     ///< @todo:  Doc
               int *iz,         ///< @todo:  Doc
               DataType *w1,    ///< @todo:  Doc
               DataType *w2,    ///< @todo:  Doc
               int *ipc,        ///< @todo:  Doc
               DataType *rpc,   ///< @todo:  Doc
               DataType *pc,    ///< @todo:  Doc
               DataType *ac,    ///< @todo:  Doc
               DataType *cc,    ///< @todo:  Doc
               DataType *fc,    ///< @todo:  Doc
               DataType *xf,    ///< @todo:  Doc
               DataType *yf,    ///< @todo:  Doc
               DataType *zf,    ///< @todo:  Doc
               DataType *gxcf,  ///< @todo:  Doc
               DataType *gycf,  ///< @todo:  Doc
               DataType *gzcf,  ///< @todo:  Doc
               DataType *a1cf,  ///< @todo:  Doc
               DataType *a2cf,  ///< @todo:  Doc
               DataType *a3cf,  ///< @todo:  Doc
               DataType *ccf,   ///< @todo:  Doc
               DataType *fcf,   ///< @todo:  Doc
               DataType *tcf,
               sycl::queue &q ///< @todo:  Doc
);

#endif /* _NEWDRVD_H_ */
