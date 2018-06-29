/* * * * *
 *  AzTimer.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_TIMER_HPP_
#define _AZ_TIMER_HPP_

#include "AzUtil.hpp"

class AzTimer {
public:
  int chk;  /* next check point */
  int inc;  /* increment: negative means no checking */

  AzTimer() : chk(-1), inc(-1) {}
  ~AzTimer() {}

  inline void reset(int inp_inc) {
    chk = -1; 
    inc = inp_inc; 
    if (inc > 0) {
      chk = inc;   
    }
  }

  inline bool ringing(bool isRinging, int inp) { /* timer is ringing */
    if (isRinging) return true; 

    if (chk > 0 && inp >= chk) {
      while(chk <= inp) {
        chk += inc; /* next check point */
      }
      return true; 
    }
    return false;
  }

  inline bool reachedMax(int inp, 
                    const char *msg, 
                    const AzOut &out) const {
    bool yes_no = reachedMax(inp); 
    if (yes_no) {
      AzTimeLog::print(msg, " reached max", out); 
    }
    return yes_no; 
  }
  inline bool reachedMax(int inp) const {
    if (chk > 0 && inp >= chk) return true; 
    else                       return false; 
  }
}; 

#endif

