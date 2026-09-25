/**
 *  @ingroup PMGC
 *  @author  Tucker Beck [fortran ->c translation], Michael Holst [original]
 *  @brief Build Galerkin matrix structures
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

#ifndef _BUILDGD_H_
#define _BUILDGD_H_

#include "apbs_macros.h"
#include <sycl/sycl.hpp>

namespace pmgc {

template <typename DataType>
void VbuildG(const int nxf,  ///< @todo: doc
             const int nyf,  ///< @todo: doc
             const int nzf,  ///< @todo: doc
             const int nxc,  ///< @todo: doc
             const int nyc,  ///< @todo: doc
             const int nzc,  ///< @todo: doc
             int *numdia,    ///< @todo: doc
             DataType *pcFF, ///< @todo: doc
             DataType *acFF, ///< @todo: doc
             DataType *ac,
             sycl::queue &q ///< @todo: doc
);

/** @brief   Computes a 27-point galerkin coarse grid matrix from
 *           a 1-point (i.e., diagonal) fine grid matrix.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *  @note    Replaces buildG_1 from buildGd.f
 *
 * Expressions for the galerkin coarse grid stencil XA in terms of
 * the fine grid matrix stencil A and the interpolation operator
 * stencil P.  these stencils have the form:
 *
 *    XA := array([
 *
 *      matrix([
 *         [ -XdNW(i,j,k), -XdN(i,j,k), -XdNE(i,j,k) ],
 *         [ -XdW(i,j,k),  -XdC(i,j,k), -XdE(i,j,k)  ],
 *         [ -XdSW(i,j,k), -XdS(i,j,k), -XdSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ -XoNW(i,j,k), -XoN(i,j,k), -XoNE(i,j,k) ],
 *         [ -XoW(i,j,k),   XoC(i,j,k), -XoE(i,j,k)  ],
 *         [ -XoSW(i,j,k), -XoS(i,j,k), -XoSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ -XuNW(i,j,k), -XuN(i,j,k), -XuNE(i,j,k) ],
 *         [ -XuW(i,j,k),  -XuC(i,j,k), -XuE(i,j,k)  ],
 *         [ -XuSW(i,j,k), -XuS(i,j,k), -XuSE(i,j,k) ]
 *      ])
 *    ]):
 *
 *    A := array([
 *
 *      matrix([
 *         [  0,           0,          0          ],
 *         [  0,           0,          0          ],
 *         [  0,           0,          0          ]
 *      ]),
 *
 *      matrix([
 *         [  0,           0,          0          ],
 *         [  0,           oC(i,j,k),  0          ],
 *         [  0,           0,          0          ]
 *      ]),
 *
 *      matrix([
 *         [  0,           0,          0          ],
 *         [  0,           0,          0          ],
 *         [  0,           0,          0          ]
 *      ])
 *
 *   P := array([
 *
 *      matrix([
 *         [ dPNW(i,j,k), dPN(i,j,k), dPNE(i,j,k) ],
 *         [ dPW(i,j,k),  dPC(i,j,k), dPE(i,j,k)  ],
 *         [ dPSW(i,j,k), dPS(i,j,k), dPSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ oPNW(i,j,k), oPN(i,j,k), oPNE(i,j,k) ],
 *         [ oPW(i,j,k),  oPC(i,j,k), oPE(i,j,k)  ],
 *         [ oPSW(i,j,k), oPS(i,j,k), oPSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ uPNW(i,j,k), uPN(i,j,k), uPNE(i,j,k) ],
 *         [ uPW(i,j,k),  uPC(i,j,k), uPE(i,j,k)  ],
 *         [ uPSW(i,j,k), uPS(i,j,k), uPSE(i,j,k) ]
 *      ])
 *    ]):
 */

template <typename DataType>
void VbuildG_1(const int nxf,  ///< @todo: doc
               const int nyf,  ///< @todo: doc
               const int nzf,  ///< @todo: doc
               const int nx,   ///< @todo: doc
               const int ny,   ///< @todo: doc
               const int nz,   ///< @todo: doc
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
               DataType *oC,   ///< @todo: doc
               DataType *XoC,  ///< @todo: doc
               DataType *XoE,  ///< @todo: doc
               DataType *XoN,  ///< @todo: doc
               DataType *XuC,  ///< @todo: doc
               DataType *XoNE, ///< @todo: doc
               DataType *XoNW, ///< @todo: doc
               DataType *XuE,  ///< @todo: doc
               DataType *XuW,  ///< @todo: doc
               DataType *XuN,  ///< @todo: doc
               DataType *XuS,  ///< @todo: doc
               DataType *XuNE, ///< @todo: doc
               DataType *XuNW, ///< @todo: doc
               DataType *XuSE, ///< @todo: doc
               DataType *XuSW,
               sycl::queue &q ///< @todo: doc
);

/** @brief   Computes a 27-point galerkin coarse grid matrix from
 *           a 7-point fine grid matrix.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *  @note    Replaces buildG_7 from buildGd.f
 *
 * Expressions for the galerkin coarse grid stencil XA in terms of
 * the fine grid matrix stencil A and the interpolation operator
 * stencil P.  these stencils have the form:
 *
 *    XA := array([
 *
 *      matrix([
 *         [ -XdNW(i,j,k), -XdN(i,j,k), -XdNE(i,j,k) ],
 *         [ -XdW(i,j,k),  -XdC(i,j,k), -XdE(i,j,k)  ],
 *         [ -XdSW(i,j,k), -XdS(i,j,k), -XdSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ -XoNW(i,j,k), -XoN(i,j,k), -XoNE(i,j,k) ],
 *         [ -XoW(i,j,k),   XoC(i,j,k), -XoE(i,j,k)  ],
 *         [ -XoSW(i,j,k), -XoS(i,j,k), -XoSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ -XuNW(i,j,k), -XuN(i,j,k), -XuNE(i,j,k) ],
 *         [ -XuW(i,j,k),  -XuC(i,j,k), -XuE(i,j,k)  ],
 *         [ -XuSW(i,j,k), -XuS(i,j,k), -XuSE(i,j,k) ]
 *      ])
 *    ]):
 *
 *    A := array([
 *
 *      matrix([
 *         [  0,           0,          0          ],
 *         [  0,          -dC(i,j,k),  0          ],
 *         [  0,           0,          0          ]
 *      ]),
 *
 *      matrix([
 *         [  0,          -oN(i,j,k),  0          ],
 *         [ -oW(i,j,k),   oC(i,j,k), -oE(i,j,k)  ],
 *         [  0,          -oS(i,j,k),  0          ]
 *      ]),
 *
 *      matrix([
 *         [  0,           0,          0          ],
 *         [  0,          -uC(i,j,k),  0          ],
 *         [  0,           0,          0          ]
 *      ])
 *
 *   P := array([
 *
 *      matrix([
 *         [ dPNW(i,j,k), dPN(i,j,k), dPNE(i,j,k) ],
 *         [ dPW(i,j,k),  dPC(i,j,k), dPE(i,j,k)  ],
 *         [ dPSW(i,j,k), dPS(i,j,k), dPSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ oPNW(i,j,k), oPN(i,j,k), oPNE(i,j,k) ],
 *         [ oPW(i,j,k),  oPC(i,j,k), oPE(i,j,k)  ],
 *         [ oPSW(i,j,k), oPS(i,j,k), oPSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ uPNW(i,j,k), uPN(i,j,k), uPNE(i,j,k) ],
 *         [ uPW(i,j,k),  uPC(i,j,k), uPE(i,j,k)  ],
 *         [ uPSW(i,j,k), uPS(i,j,k), uPSE(i,j,k) ]
 *      ])
 *    ]):
 *
 * in addition, A is assumed to be symmetric so that:
 *
 *    oS  := proc(x,y,z) RETURN( oN(x,y-1,z) ): end:
 *    oW  := proc(x,y,z) RETURN( oE(x-1,y,z) ): end:
 *    dC  := proc(x,y,z) RETURN( uC(x,y,z-1) ): end:
 *
 */

template <typename DataType>
void VbuildG_7(const int nxf,  ///< @todo: doc
               const int nyf,  ///< @todo: doc
               const int nzf,  ///< @todo: doc
               const int nx,   ///< @todo: doc
               const int ny,   ///< @todo: doc
               const int nz,   ///< @todo: doc
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
               DataType *oC,   ///< @todo: doc
               DataType *oE,   ///< @todo: doc
               DataType *oN,   ///< @todo: doc
               DataType *uC,   ///< @todo: doc
               DataType *XoC,  ///< @todo: doc
               DataType *XoE,  ///< @todo: doc
               DataType *XoN,  ///< @todo: doc
               DataType *XuC,  ///< @todo: doc
               DataType *XoNE, ///< @todo: doc
               DataType *XoNW, ///< @todo: doc
               DataType *XuE,  ///< @todo: doc
               DataType *XuW,  ///< @todo: doc
               DataType *XuN,  ///< @todo: doc
               DataType *XuS,  ///< @todo: doc
               DataType *XuNE, ///< @todo: doc
               DataType *XuNW, ///< @todo: doc
               DataType *XuSE, ///< @todo: doc
               DataType *XuSW,
               sycl::queue &q ///< @todo: doc
);

/** @brief   Compute a 27-point galerkin coarse grid matrix from
 *           a 27-point fine grid matrix.
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 *  @note    Replaces buildG_27 from buildGd.f
 *
 * Expressions for the galerkin coarse grid stencil XA in terms of
 * the fine grid matrix stencil A and the interpolation operator
 * stencil P.  these stencils have the form:
 *
 *    XA := array([
 *
 *      matrix([
 *         [ -XdNW(i,j,k), -XdN(i,j,k), -XdNE(i,j,k) ],
 *         [ -XdW(i,j,k),  -XdC(i,j,k), -XdE(i,j,k)  ],
 *         [ -XdSW(i,j,k), -XdS(i,j,k), -XdSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ -XoNW(i,j,k), -XoN(i,j,k), -XoNE(i,j,k) ],
 *         [ -XoW(i,j,k),   XoC(i,j,k), -XoE(i,j,k)  ],
 *         [ -XoSW(i,j,k), -XoS(i,j,k), -XoSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ -XuNW(i,j,k), -XuN(i,j,k), -XuNE(i,j,k) ],
 *         [ -XuW(i,j,k),  -XuC(i,j,k), -XuE(i,j,k)  ],
 *         [ -XuSW(i,j,k), -XuS(i,j,k), -XuSE(i,j,k) ]
 *      ])
 *    ]):
 *
 *    A := array([
 *
 *      matrix([
 *         [ -dNW(i,j,k), -dN(i,j,k), -dNE(i,j,k) ],
 *         [ -dW(i,j,k),  -dC(i,j,k), -dE(i,j,k)  ],
 *         [ -dSW(i,j,k), -dS(i,j,k), -dSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ -oNW(i,j,k), -oN(i,j,k), -oNE(i,j,k) ],
 *         [ -oW(i,j,k),   oC(i,j,k), -oE(i,j,k)  ],
 *         [ -oSW(i,j,k), -oS(i,j,k), -oSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ -uNW(i,j,k), -uN(i,j,k), -uNE(i,j,k) ],
 *         [ -uW(i,j,k),  -uC(i,j,k), -uE(i,j,k)  ],
 *         [ -uSW(i,j,k), -uS(i,j,k), -uSE(i,j,k) ]
 *      ])
 *    ]):
 *
 *   P := array([
 *
 *      matrix([
 *         [ dPNW(i,j,k), dPN(i,j,k), dPNE(i,j,k) ],
 *         [ dPW(i,j,k),  dPC(i,j,k), dPE(i,j,k)  ],
 *         [ dPSW(i,j,k), dPS(i,j,k), dPSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ oPNW(i,j,k), oPN(i,j,k), oPNE(i,j,k) ],
 *         [ oPW(i,j,k),  oPC(i,j,k), oPE(i,j,k)  ],
 *         [ oPSW(i,j,k), oPS(i,j,k), oPSE(i,j,k) ]
 *      ]),
 *
 *      matrix([
 *         [ uPNW(i,j,k), uPN(i,j,k), uPNE(i,j,k) ],
 *         [ uPW(i,j,k),  uPC(i,j,k), uPE(i,j,k)  ],
 *         [ uPSW(i,j,k), uPS(i,j,k), uPSE(i,j,k) ]
 *      ])
 *    ]):
 *
 * in addition, A is assumed to be symmetric so that:
 *
 *    oS  := proc(x,y,z) RETURN( oN(x,y-1,z) ): end:
 *    oW  := proc(x,y,z) RETURN( oE(x-1,y,z) ): end:
 *    oSE := proc(x,y,z) RETURN( oNW(x+1,y-1,z) ): end:
 *    oSW := proc(x,y,z) RETURN( oNE(x-1,y-1,z) ): end:
 *
 *    dC  := proc(x,y,z) RETURN( uC(x,y,z-1) ): end:
 *    dW  := proc(x,y,z) RETURN( uE(x-1,y,z-1) ): end:
 *    dE  := proc(x,y,z) RETURN( uW(x+1,y,z-1) ): end:
 *
 *    dN  := proc(x,y,z) RETURN( uS(x,y+1,z-1) ): end:
 *    dNW := proc(x,y,z) RETURN( uSE(x-1,y+1,z-1) ): end:
 *    dNE := proc(x,y,z) RETURN( uSW(x+1,y+1,z-1) ): end:
 *
 *    dS  := proc(x,y,z) RETURN( uN(x,y-1,z-1) ): end:
 *    dSW := proc(x,y,z) RETURN( uNE(x-1,y-1,z-1) ): end:
 *    dSE := proc(x,y,z) RETURN( uNW(x+1,y-1,z-1) ): end:
 */

template <typename DataType>
void VbuildG_27(const int nxf,  ///< @todo: doc
                const int nyf,  ///< @todo: doc
                const int nzf,  ///< @todo: doc
                const int nx,   ///< @todo: doc
                const int ny,   ///< @todo: doc
                const int nz,   ///< @todo: doc
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
                DataType *XoC,  ///< @todo: doc
                DataType *XoE,  ///< @todo: doc
                DataType *XoN,  ///< @todo: doc
                DataType *XuC,  ///< @todo: doc
                DataType *XoNE, ///< @todo: doc
                DataType *XoNW, ///< @todo: doc
                DataType *XuE,  ///< @todo: doc
                DataType *XuW,  ///< @todo: doc
                DataType *XuN,  ///< @todo: doc
                DataType *XuS,  ///< @todo: doc
                DataType *XuNE, ///< @todo: doc
                DataType *XuNW, ///< @todo: doc
                DataType *XuSE, ///< @todo: doc
                DataType *XuSW,
                sycl::queue &q ///< @todo: doc
);
} // namespace pmgc
#endif // _BUILDGD_H_
