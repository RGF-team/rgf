/* * * * *
 *  AzTET_Eval.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TET_EVAL_HPP_
#define _AZ_TET_EVAL_HPP_

#include "AzDataForTrTree.hpp"
#include "AzLoss.hpp"
#include "AzTE_ModelInfo.hpp"
#include "AzPerfResult.hpp"

//! Abstract class: interface for evaluation modules for Tree Ensemble Trainer.  
/*-------------------------------------------------------*/
class AzTET_Eval {
public: 
  virtual void reset(const AzDvect *inp_v_y, 
                     const char *perf_fn, 
                     bool inp_doAppend) = 0; 
  virtual void begin(const char *config="", 
                     AzLossType loss_type=AzLoss_None) = 0; 
  virtual void resetConfig(const char *config) = 0; 
  virtual void end() = 0; 
  virtual void evaluate(const AzDvect *v_p, const AzTE_ModelInfo *info, 
                        const char *user_str=NULL) = 0; 
  virtual bool isActive() const = 0; 
}; 
#endif 
