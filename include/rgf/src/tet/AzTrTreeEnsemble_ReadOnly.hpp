/* * * * *
 *  AzTrTreeEnsemble_ReadOnly.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TR_TREE_ENSEMBLE_READONLY_HPP_
#define _AZ_TR_TREE_ENSEMBLE_READONLY_HPP_

#include "AzTrTree_ReadOnly.hpp"
#include "AzTreeEnsemble.hpp"
#include "AzSvFeatInfo.hpp"

//! Abstract class: interface for read-only access to trainalbe tree ensemble.  
class AzTrTreeEnsemble_ReadOnly {
public:
  virtual bool usingTempFile() const { return false; }
  virtual const AzTrTree_ReadOnly *tree(int tx) const = 0; 
  virtual int leafNum() const = 0; 
  virtual int leafNum(int tx0, int tx1) const = 0; 
  virtual int size() const = 0; 
  virtual int max_size() const = 0;  
  virtual int lastIndex() const = 0; 
  virtual void copy_to(AzTreeEnsemble *out_ens, 
                       const char *config, const char *sign) const = 0; 
  virtual void show(const AzSvFeatInfo *feat, 
                    const AzOut &out, const char *header="") const = 0; 
  virtual double constant() const = 0; 
  virtual int orgdim() const = 0; 
  virtual const char *param_c_str() const = 0; 
}; 
#endif 
