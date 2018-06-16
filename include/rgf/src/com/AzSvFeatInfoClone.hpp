/* * * * *
 *  AzSvFeatInfoClone.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_SV_FEAT_INFO_CLONE_HPP_
#define _AZ_SV_FEAT_INFO_CLONE_HPP_

#include "AzSvFeatInfo.hpp"
#include "AzStrArray.hpp"

class AzSvFeatInfoClone : /* implements */ public virtual AzSvFeatInfo, 
                          /* implements */ public virtual AzStrArray 
{
protected:
  AzDataPool<AzBytArr> arr_desc; 

public: 
  AzSvFeatInfoClone() {}
  AzSvFeatInfoClone(const AzSvFeatInfo *inp) {
    reset(inp);   
  }
  AzSvFeatInfoClone(const AzStrArray *inp) {
    reset(inp);   
  }
  inline int featNum() const { 
    return arr_desc.size(); 
  }
  inline void concatDesc(int fx, AzBytArr *desc) const {
    if (fx < 0 || fx >= featNum()) {
      desc->c("?"); desc->cn(fx); desc->c("?");  
      return; 
    }
    desc->concat(arr_desc.point(fx)); 
  }
  void reset(const AzSvFeatInfo *inp) {
    int f_num = inp->featNum(); 
    arr_desc.reset(); 
    int fx; 
    for (fx = 0; fx < f_num; ++fx) {
      AzBytArr *ptr = arr_desc.new_slot(); 
      inp->desc(fx, ptr); 
    }
  }
  void reset(const AzStrArray *inp) {
    int f_num = inp->size(); 
    arr_desc.reset(); 
    int fx; 
    for (fx = 0; fx < f_num; ++fx) {
      AzBytArr *ptr = arr_desc.new_slot(); 
      ptr->reset(inp->c_str(fx)); 
    }    
  }
  void reset(int inp_f_num) {
    arr_desc.reset(); 
    int fx; 
    for (fx = 0; fx < inp_f_num; ++fx) {
      AzBytArr s("F"); 
      s.cn(fx, 3, true); /* width=3, fillWithZero */
      arr_desc.new_slot()->reset(&s); 
    }
  }

  void append(const AzSvFeatInfo *inp) {
    int fx; 
    for (fx = 0; fx < inp->featNum(); ++fx) {
      inp->desc(fx, arr_desc.new_slot()); 
    }
  }

  /*---  to implement AzStrArray  ---*/
  int size() const { return featNum(); }
  const char *c_str(int fx) const {
    if (fx < 0 || fx >= featNum()) {
      return "???"; 
    }
    return arr_desc.point(fx)->c_str(); 
  }
}; 
#endif 
