/* * * * *
 *  AzReg_TreeReg.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_REG_TREE_REG_HPP_
#define _AZ_REG_TREE_REG_HPP_

#include "AzUtil.hpp"
#include "AzDmat.hpp"
#include "AzTrTree_ReadOnly.hpp"
#include "AzRegDepth.hpp"
#include "AzParam.hpp"

class AzReg_TreeRegShared {
public:
  virtual AzDmat *share() = 0; 
  virtual bool create(const AzTrTree_ReadOnly *tree, const AzDmat *info) = 0; 
  virtual AzDmat *share(const AzTrTree_ReadOnly *tree) = 0; 
}; 

//! Default implementation of AzReg_TreeRegShared 
class AzReg_TreeRegShared_Dflt : /* implements */ public virtual AzReg_TreeRegShared
{
protected:
  AzDmat m_by_alltree; /* info not specific to individual tree */

public:
  /*---  override these to store tree-specific info  ---*/
  virtual AzDmat *share(const AzTrTree_ReadOnly *tree) { return NULL; }
  virtual bool create(const AzTrTree_ReadOnly *tree, const AzDmat *) { return false; }
  /*----------------------------------------------------*/
  virtual AzDmat *share() {
    return &m_by_alltree;
  }
}; 

//! Abstract class: interface to tree-structured regularizer 
class AzReg_TreeReg {
public:
  virtual void set_shared(AzReg_TreeRegShared *shared) {} 
  virtual void check_reg_depth(const AzRegDepth *) const {}

  virtual void reset(const AzTrTree_ReadOnly *inp_tree, 
                     const AzRegDepth *inp_reg_depth) = 0; 

  virtual void penalty_deriv(int nx, double *dr, 
                             double *ddr) = 0; 

  virtual void changeWeight(int nx, double w_diff) = 0; 

  virtual void clearFocusNode() = 0; 

  /*---  for node split  ---*/
  //! called by AzRgf_FindSplit_TR::begin 
  virtual void reset_forNewLeaf(const AzTrTree_ReadOnly *t, 
                                const AzRegDepth *rdep) = 0; 

  //! called by AzRgf_FindSplit_TR::findSplit 
  virtual void reset_forNewLeaf(int f_nx, 
                      const AzTrTree_ReadOnly *t, 
                      const AzRegDepth *rdep) = 0; 

  virtual double penalty_diff(const double leaf_w_delta[2]) const = 0; 
  virtual void penalty_deriv(double *dr, 
                             double *ddr) const = 0; 

  /*---  for maintenance  ---*/
  virtual void show(const AzOut &out, 
                    const char *header) const = 0; 
  virtual double penalty() const {
    return -1; 
  }

  /*---------------------------------------------------------*/
  virtual void resetParam(AzParam &param) = 0; 
  virtual void printParam(const AzOut &out) const = 0; 
  virtual void printHelp(AzHelp &h) const = 0; 

  virtual const char *signature() const = 0; 
  virtual const char *description() const = 0; 
}; 
#endif 

