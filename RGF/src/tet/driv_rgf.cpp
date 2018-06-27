/* * * * *
 *  driv_rgf.cpp 
 *  Copyright (C) 2011, 2012 Rie Johnson, 2018 RGF-team
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#define _AZ_MAIN_
#include "AzUtil.hpp"
#include "AzTETmain.hpp"
#include "AzRgfTrainerSel.hpp"
#include "AzTET_Eval_Dflt.hpp"
#include "AzHelp.hpp"

/*-----------------------------------------------------------------*/
void help(int argc, const char *argv[])
{
  cout << "Arguments: action  parameters" <<endl; 
  cout << "   action: "<<kw_train<<"|"<<kw_predict<<"|"<<kw_train_test<<"|"<<kw_train_predict<<"|"<<kw_features<<endl; 
  AzHelp h(log_out); 
  h.set_indent(11); 
  h.set_kw_width(17); 
  AzBytArr s_kw, s_desc; 
  s_kw.reset(kw_train); s_kw.c("      ..."); s_desc.reset(help_train); 
  h.item_noquotes(s_kw.c_str(), s_desc.c_str()); 
  s_kw.reset(kw_predict); s_kw.c("    ..."); s_desc.reset(help_predict); 
  h.item_noquotes(s_kw.c_str(), s_desc.c_str()); 
  s_kw.reset(kw_train_test); s_kw.c(" ..."); s_desc.reset(help_train_test); 
  h.item_noquotes(s_kw.c_str(), s_desc.c_str()); 
  s_kw.reset(kw_train_predict); s_kw.c(" ..."); s_desc.reset(help_train_predict); 
  h.item_noquotes(s_kw.c_str(), s_desc.c_str()); 
  s_kw.reset(kw_features); s_kw.c(" ..."); s_desc.reset(help_features); 
  h.item_noquotes(s_kw.c_str(), s_desc.c_str()); 
  cout << endl; 
  cout << "To get help on parameters, enter "<<argv[0]<<" action."<<endl; 
  cout << "For example:  "<<argv[0]<<" "<<kw_train_test<<endl; 
  cout << "              "<<argv[0]<<" "<<kw_train     <<endl; 
}

/*******************************************************************/
/*     main                                                        */
/*******************************************************************/
int main(int argc, const char *argv[]) 
{
  AzException *stat = NULL; 

  if (argc < 2) {
    help(argc, argv); 
    return -1; 
  }

  const char *action = argv[1]; 

  AzRgfTrainerSel alg_sel; 
  AzTET_Eval_Dflt eval; 
  AzTETmain driver(&alg_sel, &eval); 
  try {
    Az_check_system_(); 

    if (strcmp(action, kw_train) == 0) {
      driver.train(argv, argc); 
    }
    else if (strcmp(action, kw_predict) == 0) {
      driver.predict_single(argv, argc); 
    }
    else if (strcmp(action, kw_train_test) == 0) {
      driver.train_test(argv, argc); 
    }
    else if (strcmp(action, kw_train_predict) == 0) {
      driver.train_predict(argv, argc); 
    }
    else if (strcmp(action, kw_batch_predict) == 0) {
      driver.batch_predict(argv, argc); 
    }
    else if (strcmp(action, kw_features) == 0) {
      driver.features(argv, argc); 
    }
    else if (strcmp(action, kw_dump_model) == 0) {
      driver.dump_model(argv, argc);
    }
    else if (strcmp(action, kw_feature_importances) == 0) {
      driver.feature_importances(argv, argc);
    }
    else {
      help(argc, argv); 
      return -1; 
    }
  }
  catch (AzException *e) {
    stat = e; 
  }

  if (stat != NULL) {
    cout << stat->getMessage() << endl; 
    return -1; 
  }

  return 0; 
}
