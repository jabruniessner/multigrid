/**
 *  @ingroup PMGC
 *  @author  Tucker Beck [fortran ->c translation], Michael Holst [original]
 *  @brief Builds prolongation matrix
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

#ifndef _BUILDPD_H_
#define _BUILDPD_H_

#include "apbs_macros.h"
#include <sycl/sycl.hpp>

namespace pmgc {

template <typename DataType>
void VbuildP(int *nxf,      ///< @todo: doc
             int *nyf,      ///< @todo: doc
             int *nzf,      ///< @todo: doc
             int *nxc,      ///< @todo: doc
             int *nyc,      ///< @todo: doc
             int *nzc,      ///< @todo: doc
             int *mgprol,   ///< @todo: doc
             int *ipc,      ///< @todo: doc
             DataType *rpc, ///< @todo: doc
             DataType *pc,  ///< @todo: doc
             DataType *ac,  ///< @todo: doc
             DataType *xf,  ///< @todo: doc
             DataType *yf,  ///< @todo: doc
             DataType *zf,  ///< @todo: doc
             sycl::queue &q);

template <typename DataType>
void VbuildP_trilin(int *nxf,      ///< @todo: doc
                    int *nyf,      ///< @todo: doc
                    int *nzf,      ///< @todo: doc
                    int *nxc,      ///< @todo: doc
                    int *nyc,      ///< @todo: doc
                    int *nzc,      ///< @todo: doc
                    DataType *pc,  ///< @todo: doc
                    DataType *xf,  ///< @todo: doc
                    DataType *yf,  ///< @todo: doc
                    DataType *zf,  ///< @todo: doc
                    sycl::queue &q ///
);

template <typename DataType>
void VbuildPb_trilin(int *nxf,       ///< @todo: doc
                     int *nyf,       ///< @todo: doc
                     int *nzf,       ///< @todo: doc
                     int *nxc,       ///< @todo: doc
                     int *nyc,       ///< @todo: doc
                     int *nzc,       ///< @todo: doc
                     DataType *oPC,  ///< @todo: doc
                     DataType *oPN,  ///< @todo: doc
                     DataType *oPS,  ///< @todo: doc
                     DataType *oPE,  ///< @todo: doc
                     DataType *oPW,  ///< @todo: doc
                     DataType *oPNE, ///< @todo: doc
                     DataType *oPNW, ///< @todo: doc
                     DataType *oPSE, ///< @todo: doc
                     DataType *oPSW, ///< @todo: doc
                     DataType *uPC,  ///< @todo: doc
                     DataType *uPN,  ///< @todo: doc
                     DataType *uPS,  ///< @todo: doc
                     DataType *uPE,  ///< @todo: doc
                     DataType *uPW,  ///< @todo: doc
                     DataType *uPNE, ///< @todo: doc
                     DataType *uPNW, ///< @todo: doc
                     DataType *uPSE, ///< @todo: doc
                     DataType *uPSW, ///< @todo: doc
                     DataType *dPC,  ///< @todo: doc
                     DataType *dPN,  ///< @todo: doc
                     DataType *dPS,  ///< @todo: doc
                     DataType *dPE,  ///< @todo: doc
                     DataType *dPW,  ///< @todo: doc
                     DataType *dPNE, ///< @todo: doc
                     DataType *dPNW, ///< @todo: doc
                     DataType *dPSE, ///< @todo: doc
                     DataType *dPSW, ///< @todo: doc
                     DataType *xf,   ///< @todo: doc
                     DataType *yf,   ///< @todo: doc
                     DataType *zf,
                     sycl::queue &q ///< @todo: doc
);

template <typename DataType>
void VbuildP_op7(int *nxf,      ///< @todo: doc
                 int *nyf,      ///< @todo: doc
                 int *nzf,      ///< @todo: doc
                 int *nxc,      ///< @todo: doc
                 int *nyc,      ///< @todo: doc
                 int *nzc,      ///< @todo: doc
                 int *ipc,      ///< @todo: doc
                 DataType *rpc, ///< @todo: doc
                 DataType *ac,  ///< @todo: doc
                 DataType *pc,
                 sycl::queue &q ///< @todo: doc
);

template <typename DataType>
void VbuildPb_op7(int *nxf,       ///< @todo: doc
                  int *nyf,       ///< @todo: doc
                  int *nzf,       ///< @todo: doc
                  int *nxc,       ///< @todo: doc
                  int *nyc,       ///< @todo: doc
                  int *nzc,       ///< @todo: doc
                  int *ipc,       ///< @todo: doc
                  DataType *rpc,  ///< @todo: doc
                  DataType *oC,   ///< @todo: doc
                  DataType *oE,   ///< @todo: doc
                  DataType *oN,   ///< @todo: doc
                  DataType *uC,   ///< @todo: doc
                  DataType *oPC,  ///< @todo: doc
                  DataType *oPN,  ///< @todo: doc
                  DataType *oPS,  ///< @todo: doc
                  DataType *oPE,  ///< @todo: doc
                  DataType *oPW,  ///< @todo: doc
                  DataType *oPNE, ///< @todo: doc
                  DataType *oPNW, ///< @todo: doc
                  DataType *oPSE, ///< @todo: doc
                  DataType *oPSW, ///< @todo: doc
                  DataType *uPC,  ///< @todo: doc
                  DataType *uPN,  ///< @todo: doc
                  DataType *uPS,  ///< @todo: doc
                  DataType *uPE,  ///< @todo: doc
                  DataType *uPW,  ///< @todo: doc
                  DataType *uPNE, ///< @todo: doc
                  DataType *uPNW, ///< @todo: doc
                  DataType *uPSE, ///< @todo: doc
                  DataType *uPSW, ///< @todo: doc
                  DataType *dPC,  ///< @todo: doc
                  DataType *dPN,  ///< @todo: doc
                  DataType *dPS,  ///< @todo: doc
                  DataType *dPE,  ///< @todo: doc
                  DataType *dPW,  ///< @todo: doc
                  DataType *dPNE, ///< @todo: doc
                  DataType *dPNW, ///< @todo: doc
                  DataType *dPSE, ///< @todo: doc
                  DataType *dPSW,
                  sycl::queue &q ///< @todo: doc
);

template <typename DataType>
void VbuildP_op27(int *nxf,      ///< @todo: doc
                  int *nyf,      ///< @todo: doc
                  int *nzf,      ///< @todo: doc
                  int *nxc,      ///< @todo: doc
                  int *nyc,      ///< @todo: doc
                  int *nzc,      ///< @todo: doc
                  int *ipc,      ///< @todo: doc
                  DataType *rpc, ///< @todo: doc
                  DataType *ac,  ///< @todo: doc
                  DataType *pc,
                  sycl::queue &q ///< @todo: doc
);

template <typename DataType>
void VbuildPb_op27(int *nxf,       ///< @todo: doc
                   int *nyf,       ///< @todo: doc
                   int *nzf,       ///< @todo: doc
                   int *nxc,       ///< @todo: doc
                   int *nyc,       ///< @todo: doc
                   int *nzc,       ///< @todo: doc
                   int *ipc,       ///< @todo: doc
                   DataType *rpc,  ///< @todo: doc
                   DataType *oC,   ///< @todo: doc
                   DataType *oE,   ///< @todo: doc
                   DataType *oN,   ///< @todo: doc
                   DataType *uC,   ///< @todo: doc
                   DataType *oNE,  ///< @todo: doc
                   DataType *oNW,  ///< @todo: doc
                   DataType *uE,   ///< @todo: doc
                   DataType *uW,   ///< @todo: doc
                   DataType *uN,   ///< @todo: doc
                   DataType *uS,   ///< @todo: doc
                   DataType *uNE,  ///< @todo: doc
                   DataType *uNW,  ///< @todo: doc
                   DataType *uSE,  ///< @todo: doc
                   DataType *uSW,  ///< @todo: doc
                   DataType *oPC,  ///< @todo: doc
                   DataType *oPN,  ///< @todo: doc
                   DataType *oPS,  ///< @todo: doc
                   DataType *oPE,  ///< @todo: doc
                   DataType *oPW,  ///< @todo: doc
                   DataType *oPNE, ///< @todo: doc
                   DataType *oPNW, ///< @todo: doc
                   DataType *oPSE, ///< @todo: doc
                   DataType *oPSW, ///< @todo: doc
                   DataType *uPC,  ///< @todo: doc
                   DataType *uPN,  ///< @todo: doc
                   DataType *uPS,  ///< @todo: doc
                   DataType *uPE,  ///< @todo: doc
                   DataType *uPW,  ///< @todo: doc
                   DataType *uPNE, ///< @todo: doc
                   DataType *uPNW, ///< @todo: doc
                   DataType *uPSE, ///< @todo: doc
                   DataType *uPSW, ///< @todo: doc
                   DataType *dPC,  ///< @todo: doc
                   DataType *dPN,  ///< @todo: doc
                   DataType *dPS,  ///< @todo: doc
                   DataType *dPE,  ///< @todo: doc
                   DataType *dPW,  ///< @todo: doc
                   DataType *dPNE, ///< @todo: doc
                   DataType *dPNW, ///< @todo: doc
                   DataType *dPSE, ///< @todo: doc
                   DataType *dPSW,
                   sycl::queue &q ///< @todo: doc
);

} // namespace pmgc

#endif /* _BUILDPD_H_ */
