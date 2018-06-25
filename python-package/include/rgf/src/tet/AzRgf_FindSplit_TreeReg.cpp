/* * * * *
 *  AzRgf_FindSplit_TreeReg.cpp 
 *  Copyright (C) 2011, 2012 Rie Johnson, 2018 RGF-team
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#include "AzRgf_FindSplit_TreeReg.hpp"

/*--------------------------------------------------------*/
void AzRgf_FindSplit_TreeReg::findSplit(int nx, 
                           /*---  output  ---*/
                           AzTrTsplit *best_split)
{
  if (tree->usingInternalNodes()) {
    throw new AzException("AzRgf_FindSplit_TreeReg::findSplit", 
                          "can't coexist with UseInternalNodes"); 
  }

  reg->reset_forNewLeaf(nx, tree, reg_depth); 
  dR = ddR = 0; 
  reg->penalty_deriv(&dR, &ddR); 
  AzRgf_FindSplit_Dflt::findSplit(nx, best_split); 
}

/*--------------------------------------------------------*/
double AzRgf_FindSplit_TreeReg::evalSplit(
                             const Az_forFindSplit i[2],
                             double bestP[2]) const
{
  double d[2]; /* delta */
  for (int ix = 0; ix < 2; ++ix) {
    double wrsum = i[ix].wy_sum; 
    d[ix] = (wrsum-nlam*dR)/(i[ix].w_sum+nlam*ddR); 
    bestP[ix] = p_node->weight + d[ix]; 
  }

  double penalty_diff = reg->penalty_diff(d); /* new - old */

  double gain = 2*d[0]*i[0].wy_sum - d[0]*d[0]*i[0].w_sum - nlam * penalty_diff;
  gain += 2*d[1]*i[1].wy_sum - d[1]*d[1]*i[1].w_sum - nlam * penalty_diff;

  /* "2*" b/c penalty is sum v^2/2 */

  return gain;
}
