/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

// clang-format off
// 2D conv shapes
vector<conv_param_t<2>> shapes_2d = {
  // MB, IC, OC, IH, IW, G, KH, KW, stride_h, stride_w,
  // pad_h_top, pad_w_left, pad_h_bottom, pad_w_right
  // 2D convolutions
  // regular
  /*conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3},
      {1, 1}, {1, 1, 1, 1}),
  // regular with dilation
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {3, 3},
      {1, 1}, {1, 1, 1, 1}, {2, 2}),
  // groupwise
  conv_param_t<>(1, 128, 128, {56, 56}, 32, {3, 3},
      {1, 1}, {1, 1, 1, 1}),
  // DW
  conv_param_t<>(1, 272, 272, {47, 125}, 272, {3, 3},
      {1, 1}, {1, 1, 1, 1}),
  // Pointwise
  conv_param_t<>(1, 128, 128, {56, 56}, 1, {1, 1},
      {1, 1}, {0, 0, 0, 0})*/
conv_param_t<>(1,128,128,{56,56},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,128,128,{56,56},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{47,125},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,128,128,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,128,128,{56,48},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,128,128,{48,56},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,128,128,{56,56},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(2,128,128,{56,56},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,256,{28,24},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,256,{24,28},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,256,{28,28},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(2,256,256,{28,28},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,512,512,{14,12},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,512,512,{12,14},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,512,512,{14,14},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(2,512,512,{14,14},32,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,3,64,{224,224},1,{7,7},{2,2},{3,3,3,3}),
conv_param_t<>(128,3,64,{224,224},1,{7,7},{2,2},{3,3,3,3}),
conv_param_t<>(1,64,64,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,64,64,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,64,64,{56,56},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(128,64,64,{56,56},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,64,256,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,64,256,{56,56},1,{1,1},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,64,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,256,64,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,128,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,256,128,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,128,128,{56,56},1,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(128,128,128,{56,56},1,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,128,512,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,128,512,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,512,{56,56},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(128,256,512,{56,56},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,512,128,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,512,128,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,128,128,{28,28},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(128,128,128,{28,28},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,512,256,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,512,256,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,256,{28,28},1,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(128,256,256,{28,28},1,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,256,1024,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,256,1024,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,512,1024,{28,28},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(128,512,1024,{28,28},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,1024,256,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,1024,256,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,256,{14,14},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(128,256,256,{14,14},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,1024,512,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,1024,512,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,512,512,{14,14},1,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(128,512,512,{14,14},1,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,512,2048,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,512,2048,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,1024,2048,{14,14},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(128,1024,2048,{14,14},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,2048,512,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,2048,512,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,512,512,{7,7},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(128,512,512,{7,7},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,512,2048,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,512,2048,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,3,32,{224,224},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(128,3,32,{224,224},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(1,32,32,{111,111},1,{3,3},{1,1},{0,0,0,0}),
conv_param_t<>(128,32,32,{111,111},1,{3,3},{1,1},{0,0,0,0}),
conv_param_t<>(1,32,64,{109,109},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(128,32,64,{109,109},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,64,80,{109,109},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,64,80,{109,109},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,80,192,{54,54},1,{3,3},{1,1},{0,0,0,0}),
conv_param_t<>(128,80,192,{54,54},1,{3,3},{1,1},{0,0,0,0}),
conv_param_t<>(1,192,64,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,192,64,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,192,48,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,192,48,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,48,64,{25,25},1,{5,5},{1,1},{2,2,2,2}),
conv_param_t<>(128,48,64,{25,25},1,{5,5},{1,1},{2,2,2,2}),
conv_param_t<>(1,64,96,{25,25},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(128,64,96,{25,25},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,96,96,{25,25},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(128,96,96,{25,25},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,192,32,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,192,32,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,64,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,256,64,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,48,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,256,48,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,288,64,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,288,64,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,288,48,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,288,48,{25,25},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,288,384,{25,25},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(128,288,384,{25,25},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(1,96,96,{25,25},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(128,96,96,{25,25},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(1,768,192,{12,12},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,768,192,{12,12},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,768,128,{12,12},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,768,128,{12,12},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,128,128,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(128,128,128,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(1,128,192,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(128,128,192,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(1,128,128,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(128,128,128,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(1,128,192,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(128,128,192,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(1,768,160,{12,12},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,768,160,{12,12},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,160,160,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(128,160,160,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(1,160,192,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(128,160,192,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(1,160,160,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(128,160,160,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(1,160,192,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(128,160,192,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(1,192,192,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(128,192,192,{12,12},1,{1,7},{1,1},{0,0,3,3}),
conv_param_t<>(1,192,192,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(128,192,192,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(1,192,192,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(128,192,192,{12,12},1,{7,1},{1,1},{3,3,0,0}),
conv_param_t<>(1,768,128,{12,12},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,768,128,{12,12},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,128,768,{12,12},1,{5,5},{1,1},{0,0,0,0}),
conv_param_t<>(128,128,768,{12,12},1,{5,5},{1,1},{0,0,0,0}),
conv_param_t<>(1,192,320,{12,12},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(128,192,320,{12,12},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(1,192,192,{12,12},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(128,192,192,{12,12},1,{3,3},{2,2},{0,0,0,0}),
conv_param_t<>(1,1280,320,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,1280,320,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,1280,384,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,1280,384,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,384,384,{5,5},1,{1,3},{1,1},{0,0,1,1}),
conv_param_t<>(128,384,384,{5,5},1,{1,3},{1,1},{0,0,1,1}),
conv_param_t<>(1,384,384,{5,5},1,{3,1},{1,1},{1,1,0,0}),
conv_param_t<>(128,384,384,{5,5},1,{3,1},{1,1},{1,1,0,0}),
conv_param_t<>(1,1280,448,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,1280,448,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,448,384,{5,5},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(128,448,384,{5,5},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,384,384,{5,5},1,{3,1},{1,1},{1,1,0,0}),
conv_param_t<>(128,384,384,{5,5},1,{3,1},{1,1},{1,1,0,0}),
conv_param_t<>(1,1280,192,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,1280,192,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,2048,320,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,2048,320,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,2048,384,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,2048,384,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,384,384,{5,5},1,{1,3},{1,1},{0,0,1,1}),
conv_param_t<>(128,384,384,{5,5},1,{1,3},{1,1},{0,0,1,1}),
conv_param_t<>(1,2048,448,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,2048,448,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,2048,192,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,2048,192,{5,5},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,32,64,{112,112},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,32,64,{112,112},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,3,32,{224,224},1,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(128,3,32,{224,224},1,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,64,128,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,64,128,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,128,128,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,128,128,{56,56},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,128,256,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,128,256,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,256,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,256,256,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,512,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,256,512,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,512,512,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,512,512,{14,14},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,512,1024,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,512,1024,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,1024,1024,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(128,1024,1024,{7,7},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,3,64,{800,1088},1,{7,7},{2,2},{3,3,3,3}),
conv_param_t<>(1,64,64,{200,272},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,64,64,{200,272},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,64,256,{200,272},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,64,{200,272},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,128,{200,272},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,128,128,{100,136},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,128,512,{100,136},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,512,{200,272},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,512,128,{100,136},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,512,1024,{100,136},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,512,256,{100,136},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,256,256,{50,68},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,1024,{50,68},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,1024,256,{50,68},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,1024,512,{50,68},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,512,512,{25,34},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,512,2048,{25,34},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,2048,512,{25,34},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,1024,2048,{50,68},1,{1,1},{2,2},{0,0,0,0}),
conv_param_t<>(1,2048,256,{25,34},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,256,{25,34},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,512,256,{100,136},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,256,{100,136},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,256,{200,272},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,256,{200,272},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,3,{200,272},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,12,{200,272},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,3,{100,136},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,156,12,{100,136},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,3,{50,68},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,12,{50,68},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,3,{25,34},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,12,{25,34},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,256,{13,17},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,3,{13,17},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,12,{13,17},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,256,256,{14,14},1,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,256,2,{28,28},1,{1,1},{1,1},{0,0,0,0}),
conv_param_t<>(1,272,272,{47,125},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{47,125},272,{5,5},{1,1},{2,2,2,2}),
conv_param_t<>(1,272,272,{64,125},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{66,125},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{67,100},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{71,125},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{74,125},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,75},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,76},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,79},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,85},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,100},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,103},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,111},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,113},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{94,75},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{109,75},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{113,75},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,272,272,{117,75},272,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{24,63},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{32,63},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{33,63},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{34,50},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{36,63},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{37,63},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{38,38},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{38,40},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{38,43},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{38,50},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{38,52},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{38,56},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{38,57},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{47,38},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{55,38},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{57,38},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,544,544,{59,38},544,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(51,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(59,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(70,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(71,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(77,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(79,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(84,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(85,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(89,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(93,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(96,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(100,1088,1088,{7,7},1088,{3,3},{1,1},{1,1,1,1}),
conv_param_t<>(1,248,248,{93,250},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{128,250},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{132,250},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{131,250},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{133,200},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{141,250},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{148,250},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{150,150},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{150,151},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{150,158},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{150,169},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{150,200},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{150,205},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{150,221},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{150,225},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{188,150},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{218,150},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{225,150},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,248,248,{234,150},248,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{47,125},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{64,125},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{66,125},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{67,100},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{71,125},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{74,125},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,75},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,76},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,79},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,85},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,100},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,103},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,111},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{75,113},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{94,75},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{109,75},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{113,75},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,272,272,{117,75},272,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(1,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(51,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(59,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(70,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(71,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(77,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(79,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(84,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(85,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(89,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(93,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(96,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1}),
conv_param_t<>(100,544,544,{14,14},544,{3,3},{2,2},{1,1,1,1})

};

// 3D conv shapes
vector<conv_param_t<3>> shapes_3d = {
  // MB, IC, OC, {IT, IH, IW}, G, {KT, KH, KW}, {stride_t, stride_h,
  // stride_w},
  // {pad_prev, pad_h_top, pad_w_left, pad_next, pad_h_bottom, pad_w_right}
  // Regular
/*  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 1, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),

  //With dilations
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 1, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}, {2, 2, 2}),

  // Groupwise
  conv_param_t<3>(32, 192, 192, {2, 28, 28}, 96, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 192, 192, {1, 14, 14}, 96, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 384, 384, {1, 14, 14}, 192, {3, 3, 3},
      {1, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 384, 384, {1, 7, 7}, 192, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),

  conv_param_t<3>(32, 16, 16, {4, 56, 56}, 8, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 16, 16, {2, 28, 28}, 8, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 32, 32, {4, 56, 56}, 16, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 32, 32, {2, 28, 28}, 16, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 32, 32, {2, 28, 28}, 16, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 32, 32, {1, 14, 14}, 16, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 128, 128, {2, 28, 28}, 32, {3, 3, 3},
      {2, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 128, 128, {1, 14, 14}, 32, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 256, 256, {1, 14, 14}, 64, {3, 3, 3},
      {1, 2, 2}, {1, 1, 1, 1, 1, 1}),
  conv_param_t<3>(32, 256, 256, {1, 7, 7}, 64, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),

  // Depthwise
  conv_param_t<3>(1, 64, 64, {8, 14, 14}, 64, {3, 3, 3},
      {1, 1, 1}, {1, 1, 1, 1, 1, 1}),

  // Pointwise
  conv_param_t<3>(1, 128, 128, {8, 14, 14}, 1, {1, 1, 1},
      {1, 1, 1}, {0, 0, 0, 0})*/
conv_param_t<3>(1,128,128,{16,28,28},128,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(1,256,256,{8,14,14},256,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(1,512,512,{4,7,7},512,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(1,128,128,{32,56,56},128,{3,3,3},{2,2,2},{1,1,1,1,1,1}),
conv_param_t<3>(1,256,256,{16,28,28},256,{3,3,3},{2,2,2},{1,1,1,1,1,1}),
conv_param_t<3>(1,512,512,{8,14,14},512,{3,3,3},{2,2,2},{1,1,1,1,1,1}),
conv_param_t<3>(5,64,64,{32,56,56},64,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(5,128,128,{16,28,28},128,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(5,256,256,{8,14,14},256,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(5,512,512,{4,7,7},512,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(5,128,128,{32,56,56},128,{3,3,3},{2,2,2},{1,1,1,1,1,1}),
conv_param_t<3>(5,256,256,{16,28,28},256,{3,3,3},{2,2,2},{1,1,1,1,1,1}),
conv_param_t<3>(5,512,512,{8,14,14},512,{3,3,3},{2,2,2},{1,1,1,1,1,1}),
conv_param_t<3>(1,8,8,{4,4,4},8,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(1,64,64,{8,14,14},64,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(1,64,64,{8,14,14},1,{3,3,3},{1,1,1},{1,1,1,1,1,1}),
conv_param_t<3>(1,64,64,{8,14,14},1,{3,3,3},{1,1,1},{1,1,1,1,1,1})

};
// clang-format on

template <int SPATIAL_DIM, typename Acc_t>
void performance_test(const vector<conv_param_t<SPATIAL_DIM>>& shapes) {
  bool flush = true;
  std::vector<char> llc;

  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 10;

  string header = "MB, IC, OC, ";
  if (SPATIAL_DIM == 3) {
    header += "IT, ";
  }
  header += "IH, IW, G, ";
  if (SPATIAL_DIM == 3) {
    header += "KT, ";
  }
  header += "KH, KW, ";
  if (SPATIAL_DIM == 3) {
    header += "stride_t, ";
  }
  header += "stride_h, stride_w, ";
  if (SPATIAL_DIM == 3) {
    header += "pad_t, ";
  }
  header += "pad_h, pad_w, ";
  if (SPATIAL_DIM == 3) {
    header += "dilation_t, ";
  }
  header += "dilation_h, dilation_w, ";

  header += "Type, M, N, K, ";

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << "WARNING: the timer may be inaccurate when used by multiple threads."
       << endl;
  cout << header << "Im2Col (ms), "
       << "Packing (ms), "
       << "Kernel (ms), "
       << "Postprocessing (ms), "
       << "fbgemmPacked (ms), "
       << "Total (ms), "
       << "GOPS" << endl;
#else
  cout << setw(6) << header << setw(5) << "GOPS" << endl;
#endif

  chrono::time_point<chrono::high_resolution_clock> begin, end;

  for (auto conv_p : shapes) {
    if (conv_p.IC % conv_p.G != 0 || conv_p.OC % conv_p.G != 0) {
      // invalid shapes
      continue;
    }
    int im_in_dim = accumulate(
        conv_p.IN_DIM.begin(), conv_p.IN_DIM.end(), 1, multiplies<int>());
    aligned_vector<uint8_t> Aint8(conv_p.MB * im_in_dim * conv_p.IC);

    int kernel_dim =
        accumulate(conv_p.K.begin(), conv_p.K.end(), 1, multiplies<int>());
    aligned_vector<int8_t> Bint8(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    aligned_vector<int8_t> Bint8_tr(
        kernel_dim * conv_p.IC * (conv_p.OC / conv_p.G));

    int im_out_dim = accumulate(
        conv_p.OUT_DIM.begin(), conv_p.OUT_DIM.end(), 1, multiplies<int>());
    aligned_vector<int32_t> Cint32_ref(conv_p.MB * im_out_dim * conv_p.OC);
    aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
    aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);
    aligned_vector<uint8_t> Cint8_fb2(Cint32_ref.size(), 0);
    aligned_vector<int32_t> Cint32_fb2(Cint32_ref.size());

    // A matrix (input activations)
    randFill<uint8_t>(Aint8, 0, 5);
    int32_t Aint8_zero_point = 4;

    // B matrix (weights)
    randFill<int8_t>(Bint8, -4, 4);
    aligned_vector<int32_t> Bint8_zero_point(1);
    randFill(Bint8_zero_point, -3, -1);

    aligned_vector<float> C_multiplier(Bint8_zero_point.size());
    randFill(C_multiplier, 0.1234f / 2, 0.1234f * 3 / 2);
    int32_t C_zero_point = 5;

    // reference implementation
    // conv_ref expects weights to be in G (R S C/G) K/G
    transposeConvWeights<SPATIAL_DIM>(conv_p, Bint8.data(), Bint8_tr.data());
    conv_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8_tr.data(),
        Cint32_ref.data());

    // matrix dimensions after im2col
    int MDim = conv_p.MB * im_out_dim;
    int NDim = conv_p.OC / conv_p.G;
    int KDim = kernel_dim * conv_p.IC;
    int KDimPerGroup = KDim / conv_p.G;

    int OC_per_G = conv_p.OC / conv_p.G;

    // computing row offset
    vector<int32_t> row_offsets(MDim);
    vector<uint8_t> Aint8_im2col(MDim * KDim);
    im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

    // computing column offset
    vector<int32_t> col_offsets(conv_p.OC);
    for (int g = 0; g < conv_p.G; ++g) {
      col_offsets_with_zero_pt_s8acc32_ref(
          KDimPerGroup,
          OC_per_G,
          OC_per_G,
          Bint8_tr.data() + g * KDimPerGroup * OC_per_G,
          Bint8_zero_point.data(),
          col_offsets.data() + g * OC_per_G,
          conv_p.OC);
    }

    for (int g = 0; g < conv_p.G; ++g) {
      row_offsets_u8acc32_ref(
          MDim,
          KDimPerGroup,
          KDim,
          Aint8_im2col.data() + g * KDimPerGroup,
          row_offsets.data());

      requantize_u8acc32_ref(
          MDim,
          NDim,
          conv_p.G * NDim,
          Cint32_ref.data() + g * NDim,
          Cint8_ref.data() + g * NDim,
          C_multiplier.data() + g * NDim / conv_p.OC,
          C_zero_point,
          Aint8_zero_point,
          Bint8_zero_point.data() + g * NDim / conv_p.OC,
          row_offsets.data(),
          col_offsets.data() + g * NDim,
          nullptr,
          conv_p.OC);
    }

    double nops = 2.0 * static_cast<double>(NITER) * MDim * NDim * KDim;
    double ttot = 0.0;
    string runType;

    PackWeightsForConv<SPATIAL_DIM> packedB(conv_p, Bint8.data());

    runType = "UniConv";
    ttot = 0;
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    double im2col_time = 0.0;
    double total_im2col_time = 0.0;
    double total_packing_time = 0.0;
    double total_computing_time = 0.0;
    double total_kernel_time = 0.0;
    double total_postprocessing_time = 0.0;
    double total_run_time = 0.0;
#endif
    for (auto i = 0; i < NWARMUP + NITER; ++i) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      packing_time = 0.0;
      computing_time = 0.0;
      kernel_time = 0.0;
      postprocessing_time = 0.0;
      run_time = 0.0;
#endif
      llc_flush(llc);
      begin = chrono::high_resolution_clock::now();
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();
        // no-op output process objects
        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false, QuantizationGranularity::TENSOR> outputProcObj(
            doNothingObj,
            C_multiplier.data(),
            C_zero_point,
            Aint8_zero_point,
            Bint8_zero_point.data(),
            nullptr, // row offsets
            col_offsets.data(),
            nullptr, // bias
            conv_p.OC,
            conv_p.G);

        fbgemmConv(
            conv_p,
            Aint8.data(),
            packedB,
            Cint8_fb.data(),
            Cint32_fb.data(),
            outputProcObj,
            tid,
            num_threads);
      }
      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - begin);
        ttot += dur.count();
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        total_packing_time += packing_time;
        total_computing_time += computing_time;
        total_kernel_time += kernel_time;
        total_postprocessing_time += postprocessing_time;
        total_run_time += run_time;
#endif
      }
    }

    cout << conv_p.MB << ", " << conv_p.IC << ", " << conv_p.OC << ", ";
    for (int i = 0; i < SPATIAL_DIM; ++i) {
      cout << conv_p.IN_DIM[i] << ", ";
    }
    cout << conv_p.G << ", ";
    for (int i = 0; i < SPATIAL_DIM; ++i) {
      cout << conv_p.K[i] << ", ";
    }
    for (int i = 0; i < SPATIAL_DIM; ++i) {
      cout << conv_p.stride[i] << ", ";
    }
    for (int i = 0; i < SPATIAL_DIM; ++i) {
      cout << conv_p.pad[i] << ", ";
    }
    for (int i = 0; i < SPATIAL_DIM; ++i) {
      cout << conv_p.dilation[i] << ", ";
    }
    cout << setw(13) << runType << ", " << setw(5) << fixed << setw(5)
         << setw(6) << MDim << ", " << setw(6) << NDim << ", " << setw(6)
         << KDim << ", ";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << fixed << setprecision(6) << setw(8) << 0 << ", "
         << total_packing_time / (double)NITER / 1e6 << ", "
         << total_kernel_time / (double)NITER / 1e6 << ", "
         << total_postprocessing_time / (double)NITER / 1e6 << ", "
         << total_run_time / (double)NITER / 1e6 << ", "
         << ttot / (double)NITER / 1e6 << ", ";
#endif
    cout << setprecision(2) << nops / ttot << endl;

    compare_buffers(
        Cint8_ref.data(),
        Cint8_fb.data(),
        MDim,
        NDim * conv_p.G,
        NDim * conv_p.G,
        5);
  } // shapes
}

int main() {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  // performance_test<int16_t>();
  performance_test<2, int32_t>(shapes_2d);
  performance_test<3, int32_t>(shapes_3d);
  return 0;
}
