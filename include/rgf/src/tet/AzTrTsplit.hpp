/* * * * *
 *  AzTrTsplit.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson, 2018 RGF-team
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TRT_SPLIT_HPP_
#define _AZ_TRT_SPLIT_HPP_

#include "AzUtil.hpp"
#include "AzTools.hpp"

//! Node split information.  
class AzTrTsplit {
public:
  int fx; 
  double border_val; 
  double gain; 
  double bestP[2];  /* le gt */
  int weighted_n_samples[2];
  AzBytArr str_desc; 

  int tx, nx; /* set only by Rgf; not used by Std */

  AzTrTsplit() : fx(-1), border_val(0), gain(0), tx(-1), nx(-1) {
    bestP[0] = bestP[1] = 0;
    weighted_n_samples[0] = weighted_n_samples[1] = 0;
  }

  virtual void print(const char *header) {
#if 0 
printf("%s, fx=%d,border_val=%e,gain=%e,bestP[0]=%e,bestP[1]=%e,tx=%d,nx=%d\n", 
header, fx, border_val, gain, bestP[0], bestP[1], tx, nx); 
#endif
  }

  virtual 
  void reset() {
    fx = -1;
    border_val = 0;
    bestP[0] = bestP[1] = 0;
    weighted_n_samples[0] = weighted_n_samples[1] = 0;
    gain = 0;
    str_desc.reset();
    tx = nx = -1;
  }
  AzTrTsplit(int fx, double border_val, 
             double gain, 
             double bestP_L, double bestP_G) {
    reset_values(fx, border_val, gain, bestP_L, bestP_G);
  }
  AzTrTsplit(const AzTrTsplit *inp) { /* copy */
    copy(inp); 
  }
  virtual 
  inline bool isEmpty() const {
    if (fx < 0) return true; 
    return false; 
  }

  virtual 
  inline void reset(const AzTrTsplit *inp) {
    copy(inp); 
  }
  virtual 
  inline void reset(const AzTrTsplit *inp, int inp_tx, int inp_nx) {
    reset(inp); 
    tx = inp_tx; 
    nx = inp_nx; 
  }
  virtual 
  void copy(const AzTrTsplit *inp) {
    if (inp == NULL) return;
    fx = inp->fx;
    border_val = inp->border_val;
    gain = inp->gain;
    str_desc.clear();
    str_desc.concat(&inp->str_desc);
    bestP[0] = inp->bestP[0];
    bestP[1] = inp->bestP[1];
    tx = inp->tx;
    nx = inp->nx;
  }

  virtual 
  void reset_values(int inp_fx, double inp_border_val, 
                    double inp_gain,
                    double bestP_L, double bestP_G)
  {
    fx = inp_fx;
    border_val = inp_border_val;
    gain = inp_gain;
    bestP[0] = bestP_L;
    bestP[1] = bestP_G;
    tx = nx = -1; 
  }
  virtual 
  void release() {
    str_desc.reset(); 
  }
}; 
#endif 
