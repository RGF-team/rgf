/* * * * *
 *  AzTETselector.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TET_SELECTOR_HPP_
#define _AZ_TET_SELECTOR_HPP_

#include "AzTETrainer.hpp"

class AzTETselector {
public:
  //! Return trainer 
  virtual AzTETrainer *select(const char *alg_name, //! algorithm name
                              //! if true, don't throw exception on error
                              bool dontThrow=false
                              ) const = 0; 

  virtual const char *dflt_name() const = 0; 
  virtual const char *another_name() const = 0; 
  virtual const AzStrArray *names() const = 0; 
  virtual bool isRGFfamily(const char *name) const {
    AzBytArr s(name); 
    return s.beginsWith("RGF"); 
  }
  virtual bool isGBfamily(const char *name) const {
    AzBytArr s(name); 
    return s.beginsWith("GB"); 
  }

  //! Return algorithm names. 
  virtual void printOptions(const char *dlm, //! delimiter between algorithm names.  
                            AzBytArr *s) //!< output: algorithm names separated by dlm. 
                            const = 0; 

  //! Help 
  virtual void printHelp(AzHelp &h) const = 0; 
}; 

#endif 

