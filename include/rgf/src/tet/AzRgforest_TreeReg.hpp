/* * * * *
 *  AzRgforest_TreeReg.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_RGFOREST_TREEREG_HPP_
#define _AZ_RGFOREST_TREEREG_HPP_

#include "AzRgforest.hpp"

#include "AzRgf_Optimizer_TreeReg.hpp"
#include "AzRgf_FindSplit_TreeReg.hpp"
#include "AzReg_TreeRegArrImp.hpp"

//! RGF with min-penalty regularization.  

template<class T>
class AzRgforest_TreeReg : /* implements */ public virtual AzRgforest {
protected: 
  AzRgf_FindSplit_TreeReg tr_fs; 
  AzRgf_Optimizer_TreeReg tr_opt;  /* weight optimizer */
  AzReg_TreeRegArrImp<T> reg_arr; 
  AzBytArr s_sign, s_desc; 

public:
  AzRgforest_TreeReg() 
  {
    tr_fs.reset(&reg_arr); 
    tr_opt.reset(&reg_arr); 
    opt = &tr_opt; 
    fs = &tr_fs; 
    s_sign.reset(reg_arr.tmpl()->signature()); 
    s_desc.reset(reg_arr.tmpl()->description()); 
    reg_depth->set_default_for_min_penalty(); 
  }
  virtual inline const char *signature() const {
    return s_sign.c_str(); 
  }
  virtual inline const char *description() const {
    return s_desc.c_str(); 
  }

  virtual void printHelp(AzHelp &h) const {
    AzRgforest::printHelp(h); 
    reg_arr.tmpl()->printHelp(h); 
    h.begin(Azforest_config, "AzRgforest_TreeReg", "For min-penalty regularization"); 
    h.item(kw_doApproxTsr, help_doApproxTsr); 
    h.end();
  }

protected:
  virtual int resetParam(AzParam &param) { /* returns max #tree */
    int max_tree_num = AzRgforest::resetParam(param); 
    
    bool doApproxTsr = false; 
    param.swOn(&doApproxTsr, kw_doApproxTsr);  
    if (doApproxTsr) {
      AzPrint o(out); 
      o.ppBegin("AzRgforest_TreeReg", "Approximation", ", "); 
      o.printSw(kw_doApproxTsr, doApproxTsr); 
      o.ppEnd(); 
      if (doForceToRefreshAll) {
        doForceToRefreshAll = false; 
        AzPrint::writeln(out, "Turning off ", kw_doForceToRefreshAll); 
      }
    }
    else {
      if (!doForceToRefreshAll) {
        doForceToRefreshAll = true; 
        AzPrint::writeln(out, "Turning on ", kw_doForceToRefreshAll); 
      }
    }

    reg_arr.tmpl_u()->resetParam(param); 
    reg_arr.tmpl()->printParam(out);   
    reg_arr.reset(max_tree_num); 
    return max_tree_num; 
  }

  virtual void end_of_initialization() {
    AzRgforest::end_of_initialization(); 
    reg_arr.tmpl()->check_reg_depth(reg_depth);     
  }
}; 
#endif

