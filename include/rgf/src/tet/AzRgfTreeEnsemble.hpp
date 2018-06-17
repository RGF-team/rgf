/* * * * *
 *  AzRgfTreeEnsemble.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_RGF_TREE_ENSEMBLE_HPP_
#define _AZ_RGF_TREE_ENSEMBLE_HPP_

#include "AzTrTreeEnsemble_ReadOnly.hpp"
#include "AzRgfTree.hpp"
#include "AzParam.hpp"
#include "AzHelp.hpp"

//! Abstract class: interface for ensemble of RGF-trees.  
/**
  *  implemented by AzRgfTreeEnsImp<T> 
 **/  
class AzRgfTreeEnsemble : /* extends */ public virtual AzTrTreeEnsemble_ReadOnly 
{
public:
  virtual void set_constant(double inp) = 0; 
  virtual AzRgfTree *new_tree(int *out_tx=NULL) = 0; 
  virtual AzRgfTree *tree_u(int tx) const = 0; 

  virtual int nextIndex() const = 0; 
  virtual bool isFull() const = 0; 

  virtual void copy_nodes_from(const AzTrTreeEnsemble_ReadOnly *inp) = 0; 
  virtual void printHelp(AzHelp &h) const = 0; 

  virtual void cold_start(AzParam &param, 
                          const AzBytArr *s_temp_prefix, /* may be NULL */
                          int data_num, 
                          const AzOut &out, 
                          int tree_num_max, 
                          int inp_org_dim) = 0; 
  virtual void warm_start(const AzTreeEnsemble *inp_ens, 
              const AzDataForTrTree *data, 
              AzParam &param,           
              const AzBytArr *s_temp_prefix, /* may be NULL */
              const AzOut &out, 
              int max_t_num, 
              int search_t_num, 
              AzDvect *v_p, /* inout */
              const AzIntArr *inp_ia_tr_dx=NULL) = 0; 
}; 
#endif 
