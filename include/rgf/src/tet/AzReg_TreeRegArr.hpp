/* * * * *
 *  AzReg_TreeRegArr.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_REG_TREE_REG_ARR_HPP_
#define _AZ_REG_TREE_REG_ARR_HPP_

#include "AzUtil.hpp"
#include "AzReg_TreeReg.hpp"

class AzReg_TreeRegArr
{
public:
  virtual void reset(int tree_num) = 0; 
  virtual AzReg_TreeReg *reg(int tx) = 0; 
  virtual AzReg_TreeReg *reg_forNewLeaf(int tx) = 0; 
  virtual int size() const = 0; 
}; 
#endif 
