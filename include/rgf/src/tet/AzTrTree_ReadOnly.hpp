/* * * * *
 *  AzTrTree_ReadOnly.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TR_TREE_READONLY_HPP_
#define _AZ_TR_TREE_READONLY_HPP_

#include "AzUtil.hpp"
#include "AzDataForTrTree.hpp"
#include "AzTreeRule.hpp"
#include "AzSvFeatInfo.hpp"
#include "AzTreeNodes.hpp"
#include "AzTrTreeNode.hpp"
#include "AzSortedFeat.hpp"

//! Abstract class: interface for read-only (information-seeking) access to trainable tree.  
/*------------------------------------------*/
/* Trainable tree; read only */
class AzTrTree_ReadOnly : /* implements */ public virtual AzTreeNodes 
{
public:
  /*---  information seeking ... ---*/
  virtual int nodeNum() const = 0; 
  virtual int countLeafNum() const = 0; 
  virtual int maxDepth() const = 0; 
  virtual void show(const AzSvFeatInfo *feat, const AzOut &out) const = 0; 
  virtual void concat_stat(AzBytArr *o) const = 0; 
  virtual double getRule(int inp_nx, AzTreeRule *rule) const = 0; 
  virtual void concatDesc(const AzSvFeatInfo *feat, int nx, 
                  AzBytArr *str_desc, /* output */
                  int max_len=-1) const = 0; 
  virtual void isActiveNode(bool doAllowZeroWeightLeaf, 
                            AzIntArr *ia_isDecisionNode) const = 0; /* output */
  virtual bool usingInternalNodes() const = 0; 

  virtual const AzSortedFeatArr *sorted_array(int nx, 
                             const AzDataForTrTree *data) const = 0; 
                             /*--- (NOTE) this is const but changes sorted_arr[nx] ---*/

  virtual const AzIntArr *root_dx() const = 0; 

  /*---  apply ... ---*/
  virtual double apply(const AzDataForTrTree *data, int dx, 
                       AzIntArr *ia_nx=NULL) const /* node path */
                       = 0; 

  virtual const AzTrTreeNode *node(int nx) const = 0; 
}; 
#endif 
