/**
 *  @ingroup PMGC
 *  @author  Mike Holst [original], Tucker Beck [translation]
 *  @brief Build the Laplacian
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

#ifndef _VBUILDA_H_
#define _VBUILDA_H_

// #include "apbscfg.h"

#include "apbs_macros.h"
#include "precision.h"
#include <sycl/sycl.hpp>

/** @brief   Break the matrix data-structure into diagonals and then call the
 *           matrix build routine
 *  @ingroup PMGC
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 */

namespace pmgc {

template <typename DataType>
void VbuildA(int *nx,                      /**< @todo:Doc */
             int *ny,                      /**< @todo:Doc */
             int *nz,                      /**< @todo:Doc */
             int *ipkey,                   /**< @todo:Doc */
             int *mgdisc,                  /**< @todo:Doc */
             int *numdia,                  /**< @todo:Doc */
             int *ipc,                     /**< @todo:Doc */
             DataType *rpc,                /**< @todo:Doc */
             DataType *ac,                 /**< @todo:Doc */
             DataType *cc,                 /**< @todo:Doc */
             DataType *fc,                 /**< @todo:Doc */
             DataType *xf,                 /**< @todo:Doc */
             DataType *yf,                 /**< @todo:Doc */
             DataType *zf,                 /**< @todo:Doc */
             DataType *gxcf,               /**< @todo:Doc */
             DataType *gycf,               /**< @todo:Doc */
             DataType *gzcf,               /**< @todo:Doc */
             DataType *a1cf,               /**< @todo:Doc */
             DataType *a2cf,               /**< @todo:Doc */
             DataType *a3cf,               /**< @todo:Doc */
             DataType *ccf,                /**< @todo:Doc */
             DataType *fcf, sycl::queue &q /**< @todo:Doc */
);

/** @brief
 *
 *   box method (finite volume) discretization of a 3d pde on a
 *   tensor product (axi-parallel) three-dimensional mesh.
 *
 *   this subroutine discretizes the elliptic boundary value problem:
 *
 *         lu = f, u in omega
 *          u = g, u on boundary of omega
 *
 *   the multigrid code requires the operator in the form:
 *
 *        - \nabla \cdot (a \nabla u) + b \cdot u + c u = f
 *
 *   or:
 *
 *       lu = (a11 u_x)_x + (a22 u_y)_y + (a33 u_z)_z
 *            + b1 u_x + b2 u_y + b3 u_z + c u
 *
 *   here, we consider only the case: b=b1=b2=b3=0.
 *   then we assume:
 *
 *   the tensor a=diag(a11,a22,a33) has components which are
 *          then scalar functions, a11(x,y,z),a22(x,y,z),a33(x,y,z)
 *   and the functions c(x,y,z) and f(x,y,z) are also scalar.
 *   functions.  All are allowed to be possibly discontinuous on
 *   omega (the discontinuities must be along grid lines on fine grid).
 *   the boundary function g(x,y,z) is smooth on boundary of omega.
 *
 *   we will take the following conventions:
 *   (north,south,east,west refers to x-y plane.
 *   up/down refers to the z-axis)
 *     (u(x+h_x,y,z) = u^+   (east neighbor of u(x,y,z))
 *     (u(x-h_x,y,z) = u^-   (west neighbor of u(x,y,z))
 *     (u(x,y+h_y,z) = u_+   (north neighbor of u(x,y,z))
 *     (u(x,y-h_y,z) = u_-   (south neighbor of u(x,y,z))
 *     (u(x,y,z+h_z) = u.+   (up neighbor of u(x,y,z))
 *     (u(x,y,z-h_z) = u.-   (down neighbor u(x,y,z))
 *
 *   below, we will denote:
 *         u(x+h_x,y,z)       as u^+
 *         u(x+(1/2)h_x,y,z)  as u^++
 *         u(x-h_x,y,z)       as u^+
 *         u(x-(1/2)h_x,y,z)  as u^--
 *   and similarly for u_-,u_--,u_+,u_++,u.-,u.--,u.+,u.++.
 *
 *   we use the 3d analogue of the box method (see varga, pg. 191)
 *   which results in the following difference scheme:
 *
 *   u            : [ + (a11^++ + a11^--) * (h_y*h_z/h_x)
 *                    + (a22_++ + a22_--) * (h_x*h_z/h_y)
 *                    + (a33.++ + a33.--) * (h_x*h_y/h_z)
 *                    + c * (h_x*h_y*h_z) ] u
 *   u^+ (e nbr)  : [ - (a11^++) * (h_y*h_z/h_x) ]
 *   u^_ (w nbr)  : [ - (a11^--) * (h_y*h_z/h_x) ]
 *   u_+ (n nbr)  : [ - (a22_++) * (h_x*h_z/h_y) ]
 *   u_- (s nbr)  : [ - (a22_--) * (h_x*h_z/h_y) ]
 *   u.+ (u nbr)  : [ - (a33.++) * (h_x*h_y/h_z) ]
 *   u.- (d nbr)  : [ - (a33.--) * (h_x*h_y/h_z) ]
 *   f            : [ h_x*h_y*h_z ] f
 *
 *   note: fast way to do a conditional: we wish to set (a=coef),
 *         unless we are on the (z=0)-boundary, which occurs with k=1:
 *
 *            coef = ...etc...
 *            ike  = min0(1,iabs(k-1))
 *            a (index1) = (ike)*coef
 *            b(index2) = b(index2) - (1-ike)*coef*bnd_data
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 */

template <typename DataType>
void VbuildA_fv(int *nx,                      /**< @todo:Doc */
                int *ny,                      /**< @todo:Doc */
                int *nz,                      /**< @todo:Doc */
                int *ipkey,                   /**< @todo:Doc */
                int *numdia,                  /**< @todo:Doc */
                int *ipc,                     /**< @todo:Doc */
                DataType *rpc,                /**< @todo:Doc */
                DataType *oC,                 /**< @todo:Doc */
                DataType *cc,                 /**< @todo:Doc */
                DataType *fc,                 /**< @todo:Doc */
                DataType *oE,                 /**< @todo:Doc */
                DataType *oN,                 /**< @todo:Doc */
                DataType *uC,                 /**< @todo:Doc */
                DataType *xf,                 /**< @todo:Doc */
                DataType *yf,                 /**< @todo:Doc */
                DataType *zf,                 /**< @todo:Doc */
                DataType *gxcf,               /**< @todo:Doc */
                DataType *gycf,               /**< @todo:Doc */
                DataType *gzcf,               /**< @todo:Doc */
                DataType *a1cf,               /**< @todo:Doc */
                DataType *a2cf,               /**< @todo:Doc */
                DataType *a3cf,               /**< @todo:Doc */
                DataType *ccf,                /**< @todo:Doc */
                DataType *fcf, sycl::queue &q /**< @todo:Doc */
);

/** @brief  Finite element method discretization
 *
 *    Finite element method discretization of a 3d pde on a
 *    tensor product (axi-parallel) three-dimensional mesh.
 *
 *    KEY RESTRICTION: the coefficients in the pde below must
 *                     be piecewise constant in the elements
 *                     for this discretization to be formally
 *                     correct.
 *
 *    this subroutine discretizes the elliptic boundary value problem:
 *
 *          lu = f, u in omega
 *           u = g, u on boundary of omega
 *
 *    the multigrid code requires the operator in the form:
 *
 *         - \nabla \cdot (a \nabla u) + b \cdot u + c u = f
 *
 *    or:
 *
 *        lu = (a11 u_x)_x + (a22 u_y)_y + (a33 u_z)_z
 *             + b1 u_x + b2 u_y + b3 u_z + c u
 *
 *    here, we consider only the case: b=b1=b2=b3=0.
 *    then we assume:
 *
 *    the tensor a=diag(a11,a22,a33) has components which are
 *    then scalar functions, a11(x,y,z),a22(x,y,z),a33(x,y,z)
 *    and the functions c(x,y,z) and f(x,y,z) are also scalar.
 *    functions.  All are allowed to be possibly discontinuous on
 *    omega (the discontinuities must be along grid lines on fine grid).
 *    the boundary function g(x,y,z) is smooth on boundary of omega.
 *
 *    we will take the following conventions:
 *    (north,south,east,west refers to x-y plane.
 *    up/down refers to the z-axis)
 *      (u(x+h_x,y,z) = u^+   (east neighbor of u(x,y,z))
 *      (u(x-h_x,y,z) = u^-   (west neighbor of u(x,y,z))
 *      (u(x,y+h_y,z) = u_+   (north neighbor of u(x,y,z))
 *      (u(x,y-h_y,z) = u_-   (south neighbor of u(x,y,z))
 *      (u(x,y,z+h_z) = u.+   (up neighbor of u(x,y,z))
 *      (u(x,y,z-h_z) = u.-   (down neighbor u(x,y,z))
 *
 *    below, we will denote:
 *          u(x+h_x,y,z)       as u^+
 *          u(x+(1/2)h_x,y,z)  as u^++
 *          u(x-h_x,y,z)       as u^+
 *          u(x-(1/2)h_x,y,z)  as u^--
 *    and similarly for u_-,u_--,u_+,u_++,u.-,u.--,u.+,u.++.
 *
 *    we use trilinear basis functions and hexahedral elements
 *    to perform this standard finite element discretization,
 *    which results in the following difference scheme:
 *
 *    u            : [ + (a11^++ + a11^--) * (h_y*h_z/h_x)
 *                     + (a22_++ + a22_--) * (h_x*h_z/h_y)
 *                     + (a33.++ + a33.--) * (h_x*h_y/h_z)
 *                     + c * (h_x*h_y*h_z) ] u
 *    u^+ (e nbr)  : [ - (a11^++) * (h_y*h_z/h_x) ]
 *    u^_ (w nbr)  : [ - (a11^--) * (h_y*h_z/h_x) ]
 *    u_+ (n nbr)  : [ - (a22_++) * (h_x*h_z/h_y) ]
 *    u_- (s nbr)  : [ - (a22_--) * (h_x*h_z/h_y) ]
 *    u.+ (u nbr)  : [ - (a33.++) * (h_x*h_y/h_z) ]
 *    u.- (d nbr)  : [ - (a33.--) * (h_x*h_y/h_z) ]
 *    f            : [ h_x*h_y*h_z ] f
 *
 *    note: fast way to do a conditional: we wish to set (a=coef),
 *          unless we are on the (z=0)-boundary, which occurs with k=1:
 *
 *             coef = ...etc...
 *             ike  = min0(1,iabs(k-1))
 *             a (index1) = (ike)*coef
 *             b(index2) = b(index2) - (1-ike)*coef*bnd_data
 *  @todo    Update brief description
 *  @author  Tucker Beck [C Translation], Michael Holst [Original]
 */

template <typename DataType>
void VbuildA_fe(int *nx,        /**< @todo:Doc */
                int *ny,        /**< @todo:Doc */
                int *nz,        /**< @todo:Doc */
                int *ipkey,     /**< @todo:Doc */
                int *numdia,    /**< @todo:Doc */
                int *ipc,       /**< @todo:Doc */
                DataType *rpc,  /**< @todo:Doc */
                DataType *oC,   /**< @todo:Doc */
                DataType *cc,   /**< @todo:Doc */
                DataType *fc,   /**< @todo:Doc */
                DataType *oE,   /**< @todo:Doc */
                DataType *oN,   /**< @todo:Doc */
                DataType *uC,   /**< @todo:Doc */
                DataType *oNE,  /**< @todo:Doc */
                DataType *oNW,  /**< @todo:Doc */
                DataType *uE,   /**< @todo:Doc */
                DataType *uW,   /**< @todo:Doc */
                DataType *uN,   /**< @todo:Doc */
                DataType *uS,   /**< @todo:Doc */
                DataType *uNE,  /**< @todo:Doc */
                DataType *uNW,  /**< @todo:Doc */
                DataType *uSE,  /**< @todo:Doc */
                DataType *uSW,  /**< @todo:Doc */
                DataType *xf,   /**< @todo:Doc */
                DataType *yf,   /**< @todo:Doc */
                DataType *zf,   /**< @todo:Doc */
                DataType *gxcf, /**< @todo:Doc */
                DataType *gycf, /**< @todo:Doc */
                DataType *gzcf, /**< @todo:Doc */
                DataType *a1cf, /**< @todo:Doc */
                DataType *a2cf, /**< @todo:Doc */
                DataType *a3cf, /**< @todo:Doc */
                DataType *ccf,  /**< @todo:Doc */
                DataType *fcf   /**< @todo:Doc */
);

} // namespace pmgc

#endif /* _VBUILDA_H_ */
