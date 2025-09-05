/**
 *  @ingroup PMGC
 *  @author  Tucker Beck [fortran ->c translation], Michael Holst [original]
 *  @brief
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
 * Copyright (c) 2010-2020 Battelle Memorial Institute.
 * Developed at the Pacific Northwest National Laboratory, operated by Battelle
 * Memorial Institute, Pacific Northwest Division for the U.S. Department
 * Energy. Portions Copyright (c) 2002-2010, Washington University in St. Louis.
 * Portions Copyright (c) 2002-2020, Nathan A. Baker.
 * Portions Copyright (c) 1999-2002, The Regents of the University of
 * California. Portions Copyright (c) 1995, Michael Holst. All rights reserved.
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

#include "cgd.h"
#include "precision.h"

namespace pmgc {

VPUBLIC void Vcghs(int *nx, int *ny, int *nz, int *ipc, DataType *rpc,
                   DataType *ac, DataType *cc, DataType *fc, DataType *x,
                   DataType *p, DataType *ap, DataType *r, int *itmax,
                   int *iters, DataType *errtol, DataType *omega, int *iresid,
                   int *iadjoint, sycl::queue &q) {

  DataType rsnrm, pAp, denom;
  DataType rhok1, rhok2, alpha, beta;

  // Setup for the looping
  *iters = 0;

  if (*iters >= *itmax && *iresid == 0)
    return;

  Vmresid(nx, ny, nz, ipc, rpc, ac, cc, fc, x, r, q);
  denom = Vxnrm2(nx, ny, nz, r, q);

  if (denom == 0.0)
    return;

  if (*iters >= *itmax)
    return;

  while (1) {

    // Compute/check the current stopping test
    rhok2 = Vxdot(nx, ny, nz, r, r, q);
    rsnrm = VSQRT(rhok2);

    if (rsnrm / denom <= *errtol)
      break;

    if (*iters >= *itmax)
      break;

    // Form new direction vector from old one and residual
    if (*iters == 0) {
      Vxcopy(nx, ny, nz, r, p, q);
    } else {
      beta = rhok2 / rhok1;
      alpha = 1.0 / beta;
      Vxaxpy(nx, ny, nz, &alpha, r, p, q);
      Vxscal(nx, ny, nz, &beta, p, q);
    }

    // Linear case: alpha which minimizes energy norm of error
    Vmatvec(nx, ny, nz, ipc, rpc, ac, cc, p, ap, q);
    pAp = Vxdot(nx, ny, nz, p, ap, q);
    alpha = rhok2 / pAp;

    // Save rhok2 for next iteration
    rhok1 = rhok2;

    // Update solution in direction p of length alpha
    Vxaxpy(nx, ny, nz, &alpha, p, x, q);

    // Update residual
    alpha = -alpha;
    Vxaxpy(nx, ny, nz, &alpha, ap, r, q);

    // some bookkeeping
    (*iters)++;
  }
}
} // namespace pmgc
