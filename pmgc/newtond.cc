/**
 *  @ingroup  PMGC
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
 * Copyright (c) 2010-2020 Battelle Memorial Institute. Developed at the Pacific
 * Northwest National Laboratory, operated by Battelle Memorial Institute,
 * Pacific Northwest Division for the U.S. Department Energy.  Portions
 * Copyright (c) 2002-2010, Washington University in St. Louis.  Portions
 * Copyright (c) 2002-2020, Nathan A. Baker.  Portions Copyright (c) 1999-2002,
 * The Regents of the University of California. Portions Copyright (c) 1995,
 * Michael Holst.
 * All rights reserved.
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

#include "newtond.h"

namespace pmgc {

VPUBLIC void Vfnewton(int *nx, int *ny, int *nz, DataType *x, int *iz,
                      DataType *w0, DataType *w1, DataType *w2, DataType *w3,
                      int *istop, int *itmax, int *iters, int *ierror,
                      int *nlev, int *ilev, int *nlev_real, int *mgsolv,
                      int *iok, int *iinfo, DataType *epsiln, DataType *errtol,
                      DataType *omega, int *nu1, int *nu2, int *mgsmoo,
                      DataType *cprime, DataType *rhs, DataType *xtmp, int *ipc,
                      DataType *rpc, DataType *pc, DataType *ac, DataType *cc,
                      DataType *fc, DataType *tru, sycl::queue &q) {

  int level, itmxd, nlevd, iterd, iokd;
  int nxf, nyf, nzf;
  int nxc, nyc, nzc;
  int istpd;
  int numlev;
  DataType errd;

  MAT2(iz, 50, 1);

  // Recover gridsizes ***
  nxf = *nx;
  nyf = *ny;
  nzf = *nz;

  numlev = *nlev - 1;
  Vmkcors(&numlev, &nxf, &nyf, &nzf, &nxc, &nyc, &nzc);

  // Move up grids: interpolate solution to finer, do newton
  if (*iinfo > 1) {
    printf("Starting\n");
    printf("Fine Grid Size:   (%d, %d, %d)\n", nxf, nyf, nzf);
    printf("Course Grid Size: (%d, %d, %d)\n", nxc, nyc, nzc);
  }

  for (level = *nlev_real; level <= *ilev + 1; level--) {

    // Call mv cycle
    errd = *errtol;
    itmxd = 1000;
    nlevd = *nlev_real - level + 1;
    iterd = 0;
    iokd = *iok;
    istpd = *istop;

    Vnewton(&nxc, &nyc, &nzc, x, iz, w0, w1, w2, w3, &istpd, &itmxd, &iterd,
            ierror, &nlevd, &level, nlev_real, mgsolv, &iokd, iinfo, epsiln,
            &errd, omega, nu1, nu2, mgsmoo, cprime, rhs, xtmp, ipc, rpc, pc, ac,
            cc, fc, tru, q);

    // Find new grid size ***
    numlev = 1;
    Vmkfine(&numlev, &nxc, &nyc, &nzc, &nxf, &nyf, &nzf);

    // Interpolate to next finer grid (use correct bc's)
    VinterpPMG(&nxc, &nyc, &nzc, &nxf, &nyf, &nzf, RAT(x, VAT2(iz, 1, level)),
               RAT(x, VAT2(iz, 1, level - 1)), RAT(pc, VAT2(iz, 11, level - 1)),
               q);

    /*
    Commented out fortran code.  May need to implement later
    call ninterpPMG(nxc,nyc,nzc,nxf,nyf,nzf,
        x(iz(1,level)),x(iz(1,level-1)),pc(iz(11,level-1)),
        ipc(iz(5,level-1)),rpc(iz(6,level-1)),
        ac(iz(7,level-1)),cc(iz(1,level-1)),fc(iz(1,level-1)))
     */

    // New grid size
    nxc = nxf;
    nyc = nyf;
    nzc = nzf;
  }

  // Call mv cycle
  level = *ilev;

  Vnewton(nx, ny, nz, x, iz, w0, w1, w2, w3, istop, itmax, iters, ierror, nlev,
          &level, nlev_real, mgsolv, iok, iinfo, epsiln, errtol, omega, nu1,
          nu2, mgsmoo, cprime, rhs, xtmp, ipc, rpc, pc, ac, cc, fc, tru, q);
}

VPUBLIC void Vnewton(int *nx, int *ny, int *nz, DataType *x, int *iz,
                     DataType *w0, DataType *w1, DataType *w2, DataType *w3,
                     int *istop, int *itmax, int *iters, int *ierror, int *nlev,
                     int *ilev, int *nlev_real, int *mgsolv, int *iok,
                     int *iinfo, DataType *epsiln, DataType *errtol,
                     DataType *omega, int *nu1, int *nu2, int *mgsmoo,
                     DataType *cprime, DataType *rhs, DataType *xtmp, int *ipc,
                     DataType *rpc, DataType *pc, DataType *ac, DataType *cc,
                     DataType *fc, DataType *tru, sycl::queue &q) {

  int level, lev;
  int itmax_s, iters_s, ierror_s, iok_s, iinfo_s, istop_s;
  DataType errtol_s, ord, bigc;
  DataType rsden, rsnrm, orsnrm;

  DataType xnorm_old, xnorm_new, damp, xnorm_med, xnorm_den;
  DataType rho_max, rho_min, rho_max_mod, rho_min_mod, errtol_p;
  int iter_d, itmax_d, mode, idamp, ipkey;
  int itmax_p, iters_p, iok_p, iinfo_p;

  // Utility and temproary parameters
  DataType alpha;

  MAT2(iz, 50, 1);

  // Recover level information
  level = 1;
  lev = (*ilev - 1) + level;

  // Do some i/o if requested
  if (*iinfo > 1) {
    printf("Starting: (%d, %d, %d)\n", *nx, *ny, *nz);
  }

  if (*iok != 0) {
    Vprtstp(*iok, -1, 0.0, 0.0, 0.0);
  }

  /**************************************************************
   *** note: if (iok!=0) then:  use a stopping test.          ***
   ***       else:  use just the itmax to stop iteration.     ***
   **************************************************************
   *** istop=0 most efficient (whatever it is)                ***
   *** istop=1 relative residual                              ***
   *** istop=2 rms difference of successive iterates          ***
   *** istop=3 relative true error (provided for testing)     ***
   **************************************************************/

  // Compute denominator for stopping criterion
  if (*istop == 0) {
    rsden = 1.0;
  } else if (*istop == 1) {

    // Compute initial residual with zero initial guess
    // this is analogous to the linear case where one can
    // simply take norm of rhs for a zero initial guess

    Vazeros(nx, ny, nz, w1, q);

    Vnmresid(nx, ny, nz, RAT(ipc, VAT2(iz, 5, lev)), RAT(rpc, VAT2(iz, 6, lev)),
             RAT(ac, VAT2(iz, 7, lev)), RAT(cc, VAT2(iz, 1, lev)),
             RAT(fc, VAT2(iz, 1, lev)), w1, w2, w3, q);
    rsden = Vxnrm1(nx, ny, nz, w2, q);
  } else if (*istop == 2) {
    rsden = VSQRT(*nx * *ny * *nz);
  } else if (*istop == 3) {
    rsden = Vxnrm2(nx, ny, nz, RAT(tru, VAT2(iz, 1, lev)), q);
  } else if (*istop == 4) {
    rsden = Vxnrm2(nx, ny, nz, RAT(tru, VAT2(iz, 1, lev)), q);
  } else if (*istop == 5) {
    Vnmatvec(nx, ny, nz, RAT(ipc, VAT2(iz, 5, lev)), RAT(rpc, VAT2(iz, 6, lev)),
             RAT(ac, VAT2(iz, 7, lev)), RAT(cc, VAT2(iz, 1, lev)),
             RAT(tru, VAT2(iz, 1, lev)), w1, w2, q);
    rsden = VSQRT(Vxdot(nx, ny, nz, RAT(tru, VAT2(iz, 1, lev)), w1, q));
  } else {
    printf("Bad istop value: %d\n", *istop);
  }

  if (rsden == 0.0) {
    rsden = 1.0;
    printf("%b rhs is zero", rsden != 0);
  }
  rsnrm = rsden;
  orsnrm = rsnrm;

  if (*iok != 0) {
    Vprtstp(*iok, 0, rsnrm, rsden, orsnrm);
  }

  /*********************************************************************
   *** begin newton iteration
   *********************************************************************/

  // Now compute residual with the initial guess

  Vnmresid(nx, ny, nz, RAT(ipc, VAT2(iz, 5, lev)), RAT(rpc, VAT2(iz, 6, lev)),
           RAT(ac, VAT2(iz, 7, lev)), RAT(cc, VAT2(iz, 1, lev)),
           RAT(fc, VAT2(iz, 1, lev)), RAT(x, VAT2(iz, 1, lev)), w0, w2, q);
  xnorm_old = Vxnrm1(nx, ny, nz, w0, q);
  if (*iok != 0) {
    xnorm_den = rsden;
  } else {
    xnorm_den = xnorm_old;
  }

  /*********************************************************************
   *** begin the loop
   *********************************************************************/

  // Setup for the looping
  printf("Damping enabled\n");
  idamp = 1;
  *iters = 0;

  // 30
  while (1) {

    (*iters)++;

    // Save iterate if stop test will use it on next iter
    if (*istop == 2) {
      Vxcopy(nx, ny, nz, RAT(x, VAT2(iz, 1, lev)), RAT(tru, VAT2(iz, 1, lev)),
             q);
    }

    // Compute the current jacobian system and rhs
    ipkey = VAT(ipc, 10);
    Vgetjac(nx, ny, nz, nlev_real, iz, ilev, &ipkey, x, w0, cprime, rhs, cc, pc,
            q);

    // Determine number of correct digits in current residual
    // Algorithm 5.3 in the thesis, test version (1')
    // Global-superlinear convergence
    bigc = 1.0;
    ord = 2.0;

    /* NAB 06-18-01:  If complex problems are not converging, set this to
     * machine epsilon.  This makes it use the exact jacobian rather than
     * the appropriate form (as here)
     */
    errtol_s = VMIN2((0.9 * xnorm_old), (bigc * VPOW(xnorm_old, ord)));
    printf("Using errtol_s: %f", errtol_s);

    // Do a linear multigrid solve of the newton equations
    Vazeros(nx, ny, nz, RAT(xtmp, VAT2(iz, 1, lev)), q);

    itmax_s = 1000;
    istop_s = 0;
    iters_s = 0;
    ierror_s = 0;

    // NAB 06-18-01 -- What this used to be:
    iok_s = 0;
    iinfo_s = 0;
    if ((*iinfo >= 2) && (*ilev == 1))
      iok_s = 2;

    // What it's changed to:
    if (*iinfo >= 2)
      iinfo_s = *iinfo;
    iok_s = 2;

    // End of NAB hack.

    Vmvcs(nx, ny, nz, xtmp, iz, w0, w1, w2, w3, &istop_s, &itmax_s, &iters_s,
          &ierror_s, nlev, ilev, nlev_real, mgsolv, &iok_s, &iinfo_s, epsiln,
          &errtol_s, omega, nu1, nu2, mgsmoo, ipc, rpc, pc, ac, cprime, rhs,
          tru, q);

    /**************************************************************
     *** note: rhs and cprime are now available as temp vectors ***
     **************************************************************/

    // If damping is still enabled -- doit
    if (idamp == 1) {

      // Try the correction
      Vxcopy(nx, ny, nz, RAT(x, VAT2(iz, 1, lev)), w1, q);
      damp = 1.0;
      Vxaxpy(nx, ny, nz, &damp, RAT(xtmp, VAT2(iz, 1, lev)), w1, q);

      Vnmresid(nx, ny, nz, RAT(ipc, VAT2(iz, 5, lev)),
               RAT(rpc, VAT2(iz, 6, lev)), RAT(ac, VAT2(iz, 7, lev)),
               RAT(cc, VAT2(iz, 1, lev)), RAT(fc, VAT2(iz, 1, lev)), w1, w0,
               RAT(rhs, VAT2(iz, 1, lev)), q);
      xnorm_new = Vxnrm1(nx, ny, nz, w0, q);

      // Damping is still enabled -- doit
      damp = 1.0;
      iter_d = 0;
      itmax_d = 10;
      mode = 0;

      printf("Attempting damping, relres = %f", xnorm_new / xnorm_den);

      while (iter_d < itmax_d) {
        if (mode == 0) {
          if (xnorm_new < xnorm_old) {
            mode = 1;
          }
        } else if (xnorm_new > xnorm_med) {
          break;
        }

        // Keep old soln and residual around, and its norm
        Vxcopy(nx, ny, nz, w1, w2, q);
        Vxcopy(nx, ny, nz, w0, w3, q);
        xnorm_med = xnorm_new;

        // New damped correction, residual, and its norm
        Vxcopy(nx, ny, nz, RAT(x, VAT2(iz, 1, lev)), w1, q);
        damp = damp / 2.0;
        Vxaxpy(nx, ny, nz, &damp, RAT(xtmp, VAT2(iz, 1, lev)), w1, q);

        Vnmresid(nx, ny, nz, RAT(ipc, VAT2(iz, 5, lev)),
                 RAT(rpc, VAT2(iz, 6, lev)), RAT(ac, VAT2(iz, 7, lev)),
                 RAT(cc, VAT2(iz, 1, lev)), RAT(fc, VAT2(iz, 1, lev)), w1, w0,
                 RAT(rhs, VAT2(iz, 1, lev)), q);
        xnorm_new = Vxnrm1(nx, ny, nz, w0, q);

        // Next iter...
        iter_d = iter_d + 1;
        printf("Attempting damping, relres = %f", xnorm_new / xnorm_den);
      }

      Vxcopy(nx, ny, nz, w2, RAT(x, VAT2(iz, 1, lev)), q);
      Vxcopy(nx, ny, nz, w3, w0, q);
      xnorm_new = xnorm_med;
      xnorm_old = xnorm_new;

      printf("Damping accepted, relres = %f", xnorm_new / xnorm_den);

      // Determine whether or not to disable damping
      if ((iter_d - 1) == 0) {
        printf("Damping disabled");
        idamp = 0;
      }
    } else {

      // Damping is disabled -- accept the newton step
      damp = 1.0;

      Vxaxpy(nx, ny, nz, &damp, RAT(xtmp, VAT2(iz, 1, lev)),
             RAT(x, VAT2(iz, 1, lev)), q);

      Vnmresid(nx, ny, nz, RAT(ipc, VAT2(iz, 5, lev)),
               RAT(rpc, VAT2(iz, 6, lev)), RAT(ac, VAT2(iz, 7, lev)),
               RAT(cc, VAT2(iz, 1, lev)), RAT(fc, VAT2(iz, 1, lev)),
               RAT(x, VAT2(iz, 1, lev)), w0, RAT(rhs, VAT2(iz, 1, lev)), q);

      xnorm_new = Vxnrm1(nx, ny, nz, w0, q);
      xnorm_old = xnorm_new;
    }

    // Compute/check the current stopping test ***
    if (iok != 0) {

      orsnrm = rsnrm;

      if (*istop == 0) {
        rsnrm = xnorm_new;
      } else if (*istop == 1) {
        rsnrm = xnorm_new;
      } else if (*istop == 2) {
        Vxcopy(nx, ny, nz, RAT(tru, VAT2(iz, 1, lev)), w1, q);
        alpha = -1.0;
        Vxaxpy(nx, ny, nz, &alpha, RAT(x, VAT2(iz, 1, lev)), w1, q);
        rsnrm = Vxnrm1(nx, ny, nz, w1, q);
      } else if (*istop == 3) {
        Vxcopy(nx, ny, nz, RAT(tru, VAT2(iz, 1, lev)), w1, q);
        alpha = -1.0;
        Vxaxpy(nx, ny, nz, &alpha, RAT(x, VAT2(iz, 1, lev)), w1, q);
        rsnrm = Vxnrm2(nx, ny, nz, w1, q);
      } else if (*istop == 4) {
        Vxcopy(nx, ny, nz, RAT(tru, VAT2(iz, 1, lev)), w1, q);
        alpha = -1.0;
        Vxaxpy(nx, ny, nz, &alpha, RAT(x, VAT2(iz, 1, lev)), w1, q);
        rsnrm = Vxnrm2(nx, ny, nz, w1, q);
      } else if (*istop == 5) {
        Vxcopy(nx, ny, nz, RAT(tru, VAT2(iz, 1, lev)), w1, q);
        alpha = -1.0;
        Vxaxpy(nx, ny, nz, &alpha, RAT(x, VAT2(iz, 1, lev)), w1, q);
        Vnmatvec(nx, ny, nz, RAT(ipc, VAT2(iz, 5, lev)),
                 RAT(rpc, VAT2(iz, 6, lev)), RAT(ac, VAT2(iz, 7, lev)),
                 RAT(cc, VAT2(iz, 1, lev)), w1, w2, w3, q);
        rsnrm = VSQRT(Vxdot(nx, ny, nz, w1, w2, q));
      } else {
        printf("Bad istop value: %d", *istop);
        exit(-1);
      }

      Vprtstp(*iok, *iters, rsnrm, rsden, orsnrm);

      if ((rsnrm / rsden) <= *errtol)
        break;
    }

    // Check iteration count ***
    if (*iters >= *itmax)
      break;
  }

  // Condition estimate of final jacobian
  if (*iinfo > 2) {

    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("%% Vnewton: JACOBIAN ANALYSIS ==> (%d, %d, %d)\n", *nx, *ny, *nz);

    // Largest eigenvalue of the jacobian matrix
    printf("%% Vnewton: Power calculating rho(JAC)\n");
    itmax_p = 1000;
    errtol_p = 1.0e-4;
    iters_p = 0;
    iinfo_p = *iinfo;

    Vpower(nx, ny, nz, iz, ilev, ipc, rpc, ac, cprime, w0, w1, w2, w3, &rho_max,
           &rho_max_mod, &errtol_p, &itmax_p, &iters_p, &iinfo_p, q);

    printf("%% Vnewton: power iters   = %d\n", iters_p);
    printf("%% Vnewton: power eigmax  = %.0f\n", rho_max);
    printf("%% Vnewton: power (MODEL) = %.0f\n", rho_max_mod);

    // Smallest eigenvalue of the system matrix A ***
    printf("%% Vnewton: ipower calculating lambda_min(JAC)...\n");
    itmax_p = 1000;
    errtol_p = 1.0e-4;
    iters_p = 0;
    iinfo_p = *iinfo;

    Vazeros(nx, ny, nz, xtmp, q);

    Vipower(nx, ny, nz, xtmp, iz, w0, w1, w2, w3, rhs, &rho_min, &rho_min_mod,
            &errtol_p, &itmax_p, &iters_p, nlev, ilev, nlev_real, mgsolv,
            &iok_p, &iinfo_p, epsiln, errtol, omega, nu1, nu2, mgsmoo, ipc, rpc,
            pc, ac, cprime, tru, q);

    printf("%% Vnewton: ipower iters   = %d\n", iters_p);
    printf("%% Vnewton: ipower eigmin  = %.0f\n", rho_min);
    printf("%% Vnewton: ipower (MODEL) = %.0f\n", rho_min_mod);

    // Condition number estimate
    printf("%% Vnewton: condition number  = %f\n", (DataType)rho_max / rho_min);
    printf("%% Vnewton: condition (MODEL) = %f\n",
           (DataType)rho_max_mod / rho_min_mod);
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
  }
}

VPUBLIC void Vgetjac(int *nx, int *ny, int *nz, int *nlev_real, int *iz,
                     int *lev, int *ipkey, DataType *x, DataType *r,
                     DataType *cprime, DataType *rhs, DataType *cc,
                     DataType *pc, sycl::queue &q) {

  int nxx, nyy, nzz;
  int nxold, nyold, nzold;
  int level, numlev;

  MAT2(iz, 50, 1);

  // Setup
  nxx = *nx;
  nyy = *ny;
  nzz = *nz;

  // Form the rhs of the newton system -- just current residual
  Vxcopy(nx, ny, nz, r, RAT(rhs, VAT2(iz, 1, *lev)), q);

  // Get nonlinear part of the jacobian operator
  Vdc_vec(RAT(cc, VAT2(iz, 1, *lev)), RAT(x, VAT2(iz, 1, *lev)),
          RAT(cprime, VAT2(iz, 1, *lev)), nx, ny, nz, ipkey);

  // Build the (nlev-1) level operators
  for (level = *lev + 1; level <= *nlev_real; level++) {
    nxold = nxx;
    nyold = nyy;
    nzold = nzz;

    numlev = 1;
    Vmkcors(&numlev, &nxold, &nyold, &nzold, &nxx, &nyy, &nzz);

    // Make the coarse grid rhs functions
    Vrestrc(&nxold, &nyold, &nzold, &nxx, &nyy, &nzz,
            RAT(rhs, VAT2(iz, 1, level - 1)), RAT(rhs, VAT2(iz, 1, level)),
            RAT(pc, VAT2(iz, 11, level - 1)), q);

    // Make the coarse grid helmholtz terms
    Vrestrc(&nxold, &nyold, &nzold, &nxx, &nyy, &nzz,
            RAT(cprime, VAT2(iz, 1, level - 1)),
            RAT(cprime, VAT2(iz, 1, level)), RAT(pc, VAT2(iz, 11, level - 1)),
            q);
  }
}

} // namespace pmgc
