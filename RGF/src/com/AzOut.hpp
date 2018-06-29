/* * * * *
 *  AzOut.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_OUT_HPP_
#define _AZ_OUT_HPP_

class AzOut {
protected:
  bool isActive; 
  int level; 
public:
  ostream *o; 

  inline AzOut() : o(NULL), isActive(true), level(0) {}
  inline AzOut(ostream *o_ptr) : isActive(true), level(0) {
    o = o_ptr;
  }
  inline void reset(ostream *o_ptr) {
    o = o_ptr; 
    activate(); 
  }

  inline void deactivate() { 
    isActive = false; 
  }
  inline void activate() { 
    isActive = true; 
  }
  inline void setStdout() { 
    o = &cout; 
    activate(); 
  }
  inline void setStderr() { 
    o = &cerr; 
    activate(); 
  }
  inline bool isNull() const { 
    if (!isActive) return true; 
    if (o == NULL) return true; 
    return false; 
  }
  inline void flush() const { 
    if (o != NULL) o->flush(); 
  }
  inline void setLevel(int inp_level) {
    level = inp_level; 
  }
  inline int getLevel() const {
    return level; 
  }
}; 
#endif 
