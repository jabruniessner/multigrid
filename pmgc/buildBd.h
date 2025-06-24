/**
 *  @ingroup PMGC
 *  @author Mike Holst and Steve Bond [original], Tucker Beck [translation]
 *  @brief Banded matrix builder
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

#ifndef _BUILDBD_H_
#define _BUILDBD_H_

// #include "apbscfg.h"

#include "abps_macros.h"

#include "mlinpckd.h"

/** @brief   Build and factor a banded matrix given a matrix in diagonal form.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *  @note    Replaces buildband from buildBd.f
 */

template <typename DataType>
void Vbuildband(int *key,       ///< @todo:  Doc
                int *nx,        ///< @todo:  Doc
                int *ny,        ///< @todo:  Doc
                int *nz,        ///< @todo:  Doc
                int *ipc,       ///< @todo:  Doc
                DataType *rpc,  ///< @todo:  Doc
                DataType *ac,   ///< @todo:  Doc
                int *ipcB,      ///< @todo:  Doc
                DataType *rpcB, ///< @todo:  Doc
                DataType *acB   ///< @todo:  Doc
);

/** @brief   Build the operator in banded form given the 7-diagonal form.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *  @note    Replaces buildband1_7 from buildBd.f
 */

template <typename DataType>
void Vbuildband1_7(int *nx,        ///< @todo:  Doc
                   int *ny,        ///< @todo:  Doc
                   int *nz,        ///< @todo:  Doc
                   int *ipc,       ///< @todo:  Doc
                   DataType *rpc,  ///< @todo:  Doc
                   DataType *oC,   ///< @todo:  Doc
                   DataType *oE,   ///< @todo:  Doc
                   DataType *oN,   ///< @todo:  Doc
                   DataType *uC,   ///< @todo:  Doc
                   int *ipcB,      ///< @todo:  Doc
                   DataType *rpcB, ///< @todo:  Doc
                   DataType *acB,  ///< @todo:  Doc
                   int *n,         ///< @todo:  Doc
                   int *m,         ///< @todo:  Doc
                   int *lda        ///< @todo:  Doc
);

/** @brief   Build the operator in banded form given the 27-diagonal form.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *  @note    Replaces buildband1_7 from buildBd.f
 */

template <typename DataType>
void Vbuildband1_27(int *nx,        ///< @todo:  Doc
                    int *ny,        ///< @todo:  Doc
                    int *nz,        ///< @todo:  Doc
                    int *ipc,       ///< @todo:  Doc
                    DataType *rpc,  ///< @todo:  Doc
                    DataType *oC,   ///< @todo:  Doc
                    DataType *oE,   ///< @todo:  Doc
                    DataType *oN,   ///< @todo:  Doc
                    DataType *uC,   ///< @todo:  Doc
                    DataType *oNE,  ///< @todo:  Doc
                    DataType *oNW,  ///< @todo:  Doc
                    DataType *uE,   ///< @todo:  Doc
                    DataType *uW,   ///< @todo:  Doc
                    DataType *uN,   ///< @todo:  Doc
                    DataType *uS,   ///< @todo:  Doc
                    DataType *uNE,  ///< @todo:  Doc
                    DataType *uNW,  ///< @todo:  Doc
                    DataType *uSE,  ///< @todo:  Doc
                    DataType *uSW,  ///< @todo:  Doc
                    int *ipcB,      ///< @todo:  Doc
                    DataType *rpcB, ///< @todo:  Doc
                    DataType *acB,  ///< @todo:  Doc
                    int *n,         ///< @todo:  Doc
                    int *m,         ///< @todo:  Doc
                    int *lda        ///< @todo:  Doc
);

#endif /* _BUILDBD_H_ */
