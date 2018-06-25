/* * * * *
 *  AzTrTreeNode.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson, 2018 RGF-team
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TR_TREE_NODE_HPP_
#define _AZ_TR_TREE_NODE_HPP_

#include "AzTreeNodes.hpp"

class AzTrTree; 

/*---------------------------------------------*/
/*! used only for training */
class AzTrTreeNode : /* extends */ public virtual AzTreeNode {
protected:
  const int *dxs; /* data indexes belonging to this node */

public:
  int dxs_offset;  /* position in the data indexes at the root */
  int dxs_num; 
  int depth; //!< node depth

  AzTrTreeNode() : depth(-1), dxs(NULL), dxs_offset(-1), dxs_num(-1) {}
  void reset() {
    AzTreeNode::reset();
    depth = dxs_offset = dxs_num = -1;
    dxs = NULL;
  }
  void transfer_from(AzTrTreeNode *inp) {
    AzTreeNode::transfer_from(inp);
    dxs = inp->dxs;
    dxs_offset = inp->dxs_offset;
    dxs_num = inp->dxs_num;
    depth = inp->depth;
    gain = inp->gain;
  }

  inline const int *data_indexes() const {
    if (dxs_num > 0 && dxs == NULL) {
      throw new AzException("AzTrTreeNode::data_indexes",
                            "data indexes are unavailable");
    }
    return dxs;
  }
  inline void reset_data_indexes(const int *ptr) {
    dxs = ptr;
  }

  friend class AzTrTree; 
}; 

#endif 
