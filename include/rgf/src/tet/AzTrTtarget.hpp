/* * * * *
 *  AzTrTtarget.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TRT_TARGET_HPP_
#define _AZ_TRT_TARGET_HPP_

#include "AzUtil.hpp"
#include "AzDmat.hpp"

//! Targets and data point weights for node split search.  
/*--------------------------------------------------------*/
class AzTrTtarget {
protected:
  AzDvect v_tar_dw;  // v_y - v_p(red)
  AzDvect v_dw;
  AzDvect v_y; 
  AzDvect v_fixed_dw; /* data point weights assigned by users */
  double fixed_dw_sum; 

public:
  AzTrTtarget() : fixed_dw_sum(-1) {}
  AzTrTtarget(const AzDvect *inp_v_y, 
              const AzDvect *inp_v_fixed_dw=NULL) {
    reset(inp_v_y, inp_v_fixed_dw); 
  }
  void reset(const AzDvect *inp_v_y, 
             const AzDvect *inp_v_fixed_dw=NULL) {
    v_dw.reform(inp_v_y->rowNum()); 
    v_dw.set(1); 
    v_tar_dw.set(inp_v_y); 
    v_y.set(inp_v_y);
    fixed_dw_sum = -1; 

    v_fixed_dw.reset(); 
    if (!AzDvect::isNull(inp_v_fixed_dw)) {
      v_fixed_dw.set(inp_v_fixed_dw); 
      if (v_fixed_dw.rowNum() != v_y.rowNum()) {
        throw new AzException(AzInputError, "AzTrTtarget::reset", 
                              "conlict in dimensionality: y and data point weights"); 
      }
      fixed_dw_sum = v_fixed_dw.sum(); 
    }
  }
  inline bool isWeighted() const {
    return !AzDvect::isNull(&v_fixed_dw); 
  }
  inline double sum_fixed_dw() const {
    return fixed_dw_sum; 
  }
  inline void weight_tarDw() {
    v_tar_dw.scale(&v_fixed_dw); 
  }
  inline void weight_dw() {
    v_dw.scale(&v_fixed_dw); 
  }

  AzTrTtarget(const AzTrTtarget *inp) {
    reset(inp); 
  }

  void reset(const AzTrTtarget *inp) {
    if (inp != NULL) {
      v_tar_dw.set(&inp->v_tar_dw); 
      v_dw.set(&inp->v_dw); 
      v_y.set(&inp->v_y); 
      v_fixed_dw.set(&inp->v_fixed_dw); 
      fixed_dw_sum = inp->fixed_dw_sum; 
    }
  }

  void resetTarDw_residual(const AzDvect *v_p) { /* only for LS */
    v_tar_dw.set(&v_y); 
    v_tar_dw.add(v_p, -1); 
  }
  inline const double *dw_arr() const {
    return v_dw.point(); 
  }
  inline const double *tarDw_arr() const {
    return v_tar_dw.point(); 
  }
  inline const AzDvect *y() const {
    return &v_y; 
  }
  inline int dataNum() const {
    return v_tar_dw.rowNum(); 
  }
  inline AzDvect *tarDw_forUpdate() {
    return &v_tar_dw; 
  }
  inline const AzDvect *tarDw() const {
    return &v_tar_dw; 
  }
  inline AzDvect *dw_forUpdate() {
    return &v_dw; 
  }
  inline const AzDvect *dw() {
    return &v_dw; 
  }
  inline double getTarDwSum(const int *dxs, int dxs_num) const {
    return v_tar_dw.sum(dxs, dxs_num); 
  }
  inline double getDwSum(const int *dxs, int dxs_num) const {
    return v_dw.sum(dxs, dxs_num); 
  }
  inline double getTarDwSum(const AzIntArr *ia_dx=NULL) const {
    return v_tar_dw.sum(ia_dx); 
  }
  inline double getDwSum(const AzIntArr *ia_dx=NULL) const {
    return v_dw.sum(ia_dx); 
  }

  int dim() const {
    return v_tar_dw.rowNum(); 
  }
}; 
#endif 
