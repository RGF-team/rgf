/* * * * *
 *  AzStrArray.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_STR_ARRAY_HPP_
#define _AZ_STR_ARRAY_HPP_

#include "AzUtil.hpp"

class AzStrArray {
public:
  virtual int size() const = 0; 
  virtual const char *c_str(int no) const = 0; 
  void get(int no, AzBytArr *byteq) const {
    byteq->reset(); 
    byteq->concat(c_str(no)); 
  }

  virtual bool isSame(const AzStrArray *inp) const {
    if (size() != inp->size()) {
      return false; 
    }
    int ix; 
    for (ix = 0; ix < size(); ++ix) {
      AzBytArr s0; 
      get(ix, &s0); 
      if (s0.compare(inp->c_str(ix)) != 0) {
        return false; 
      }
    }
    return true; 
  }

  /*---*/
  virtual void writeText(const char *fn) const {
    AzFile file(fn); 
    file.open("wb"); 
    int ix; 
    for (ix = 0; ix < size(); ++ix) {
      AzBytArr s; 
      get(ix, &s); 
      s.nl(); 
      s.writeText(&file); 
    }
    file.close(true); 
  }
}; 

#endif 

