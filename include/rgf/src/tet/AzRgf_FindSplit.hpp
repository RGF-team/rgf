/* * * * *
 *  AzRgf_FindSplit.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_RGF_FIND_SPLIT_HPP_
#define _AZ_RGF_FIND_SPLIT_HPP_

#include "AzTrTree_ReadOnly.hpp"
#include "AzTrTsplit.hpp"
#include "AzTrTtarget.hpp"
#include "AzRgf_kw.hpp"
#include "AzRegDepth.hpp"
#include "AzParam.hpp"
#include "AzFsinfo.hpp"
#include "AzHelp.hpp"

class AzRgf_FindSplit_input {
public:
  int tx; 
  const AzDataForTrTree *data; 
  const AzTrTtarget *target; 
  double lam_scale; /*!< for numerical stability of exp loss */
  double nn; /* sum of data point weights if weighted */

  AzRgf_FindSplit_input(int inp_tx, 
                        const AzDataForTrTree *inp_data, 
                        const AzTrTtarget *inp_target, 
                        double inp_lam_scale, 
                        double inp_nn) {
    tx = inp_tx; 
    data = inp_data; 
    target = inp_target; 
    lam_scale = inp_lam_scale; 
    nn = (double)inp_nn; 
  }
}; 

/*--------------------------------------------------------*/
//! Abstract class: interface for node split search for RGF. 
/**
  *  Implemented by AzRgf_FindSplit_Dflt, AzRgf_FindSplit_TreeReg
 **/
class AzRgf_FindSplit {
public:
  virtual void reset(AzParam &param, 
                     const AzRegDepth *reg_depth,
                     const AzOut &out) = 0; 

  virtual void begin(const AzTrTree_ReadOnly *tree, 
                   const AzRgf_FindSplit_input &inp, 
                   int min_size)
                   = 0; 
  virtual void begin(const AzTrTree_ReadOnly *tree, 
                   const AzRgf_FindSplit_input &inp, 
                   int min_size,    
                   AzFsinfoOnTree *fot) { /* added for TreeRegFast */
    throw new AzException("AzRgf_FindSplit::begin(...fot)", 
                          "no appropriate override"); 
  }

  virtual void pickFeats(int f_num, int data_num) = 0; 

  virtual void end() = 0; 
  virtual 
  void findSplit(int nx, //!< node id 
                 /*---  output  ---*/
                 AzTrTsplit *best_split) = 0; 

  virtual void printParam(const AzOut &out) const = 0;  
  virtual void printHelp(AzHelp &h) const = 0; 
}; 
#endif 


