/* * * * *
 *  AzTreeNodes.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson, 2018 RGF-team
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TREE_NODES_HPP_
#define _AZ_TREE_NODES_HPP_

#include "AzUtil.hpp"

/*! Tree node */
class AzTreeNode {
public:
  int fx; //!< feature id
  double border_val;
  int le_nx; //!< x[fx] <= border_val
  int gt_nx; //!< x[fx] >  border_val
  int parent_nx; //!< pointing parent node
  double weight; //!< weight
  double gain; //!< impurity for calc feature importances

  /*---  ---*/
  AzTreeNode() {
    reset(); 
  }
  void reset() {
    border_val = 0;
    weight = 0;
    gain = 0;
    fx = le_nx = gt_nx = parent_nx = -1; 
  }
  AzTreeNode(AzFile *file) {
    read(file); 
  }
  inline bool isLeaf() const {
    if (le_nx < 0) return true; 
    return false; 
  }
  void write(AzFile *file); 
  void read(AzFile *file); 

  void transfer_from(AzTreeNode *inp) {
    *this = *inp; 
  }
}; 

class AzTreeNodes {
public:
  virtual const AzTreeNode *node(int nx) const = 0; 
  virtual int nodeNum() const = 0; 
  virtual int root() const = 0; 
}; 
#endif 
