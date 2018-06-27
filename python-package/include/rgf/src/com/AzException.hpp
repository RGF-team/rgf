/* * * * *
 *  AzException.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_EXCEPTION_HPP_
#define _AZ_EXCEPTION_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>
using namespace std; 

enum AzRetCode {
  AzNormal=0, 
  AzAllocError=10,
  AzFileIOError=20,
  AzInputError=30, 
  AzInputMissing=31, 
  AzInputNotValid=32, 
  AzConflict=100, /* all others */
}; 

/*-----------------------------------------------------*/
class AzException {
public:
  AzException(const char *string1, 
              const char *string2, 
              const char *string3=NULL) 
  {
    reset(AzConflict, string1, string2, string3); 
  }

  AzException(AzRetCode retcode, 
              const char *string1, 
              const char *string2, 
              const char *string3=NULL) 
  {
    reset(retcode, string1, string2, string3); 
  }

  template <class T>
  AzException(AzRetCode retcode, 
              const char *string1, 
              const char *string2, 
              const char *string3, 
              T anything)
  {
    reset(retcode, string1, string2, string3); 
    s3 << "; " << anything; 
  }

  void reset(AzRetCode retcode, 
             const char *str1, 
             const char *str2, 
             const char *str3)
  {
    this->retcode = retcode; 
    if (str1 != NULL) s1 << str1; 
    if (str2 != NULL) s2 << str2; 
    if (str3 != NULL) s3 << str3; 
  }

  AzRetCode getReturnCode() {
    return retcode;   
  }

  string getMessage() 
  {
    if (retcode == AzNormal) {

    }
    else if (retcode == AzAllocError) {
      message << "!Memory alloc error!"; 
    }
    else if (retcode == AzFileIOError) {
      message << "!File I/O error!"; 
    }
    else if (retcode == AzInputError) {
      message << "!Input error!"; 
    }
    else if (retcode == AzInputMissing) {
      message << "!Missing input!"; 
    }
    else if (retcode == AzInputNotValid) {
      message << "!Input value is not valid!"; 
    }
    else if (retcode == AzConflict) {
      message << "Conflict"; 
    }
    else { 
      message << "Unknown error"; 
    }

    message << ": "; 
    if (s1.str().find("Az") == 0) {
      message << "(Detected in " << s1.str() << ") " << endl; 
    }
    else {
      message << s1.str() << " "; 
    }
    message << s2.str(); 
    if (s3.str().length() > 0) {
      message << " " << s3.str(); 
    }
    message << endl; 
    return message.str(); 
  }

protected:
  AzRetCode retcode; 

  stringstream s1, s2, s3; 
  stringstream message; 
};

#endif 
