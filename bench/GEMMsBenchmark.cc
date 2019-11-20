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
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_MKLDNN
#include <mkldnn.h>
#include "gemm_pack.hpp"
#endif

#include "BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"
#include "test/QuantizationHelpers.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  // clang-format off
  static const vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // m, n, k
    {64, 800, 320},
    {64, 768, 512},
    {16, 256, 512},
    {128, 128, 128},
    {256, 512, 256},
    {1024, 1024, 1024},

    {156800, 4, 36},
    {156800, 8, 36},
    {156800, 16, 36},
    {1, 128, 512},
    {1, 1024, 256},
    {1, 2048, 512},
    {1, 4096, 1024},
    {6, 256, 1024},
    {6, 256, 2048},
    {6, 512, 512},
    {6, 1024, 256},
    {6, 2048, 256},
    {6, 2048, 512},
    {6, 4096, 256},
    {6, 4096, 1024},
    {6, 4096, 2048},

    {10, 2048, 256},
    {10, 4096, 1024},

    {20, 2048, 256},
    {20, 4096, 1024},

    {102, 1024, 512},
    {102, 2323, 256},
    {102, 512, 256},

    {1, 800, 3200},
    {1, 800, 8000},

    {16, 256, 1500},
    {16, 256, 1567},
    {1, 128, 2876},
    {16, 128, 1567},
    {1, 128, 2722},
#if 0
    // benchdnn sizes
    {1024, 1024, 1024},
    {128, 128, 128},
    {15, 6400, 141},
    {211, 16, 2504},
    {256, 16, 512},
    {369, 16, 1434},
    {512, 256, 256},
    {768, 64, 512},
    {800, 64, 320},
    {8, 6400, 141},
 #else
    // FB sizes, mkldnn patch
    {6400, 15, 141},
    {6400, 8, 141},
    {16, 211, 2504},
    {16, 369, 1434},
    {1, 1024, 3496},
    {16, 256, 512},
    {1, 1600, 3456},
    // Inception_v3
    {1, 2048, 1000},
    // mask-rcnn-R-50-FPN-x1
    {1000, 1024, 12544},
    {1000, 1024, 1024},
    {1000, 2, 1024},
    {1000, 8, 1024},
    // GEMMsTunableBenchmark
#endif
  };
  // clang-format on
  bool flush = true;
  std::vector<char> llc;

  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 10;

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  cout << "WARNING: the timer may be inaccurate when used by multiple threads."
       << endl;
  cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
       << "Type, " << setw(18) << "Packing (us), " << setw(18)
       << "Kernel (us), " << setw(18) << "Postproc (us), " << setw(18)
       << "Total (us), " << setw(5) << "GOPs" << endl;
#else
  cout << setw(8) << "M, " << setw(8) << "N, " << setw(8) << "K, " << setw(18)
       << "Type, " << setw(5) << "GOPS" << endl;
#endif

  chrono::time_point<chrono::high_resolution_clock> start, end;
  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];
    const long int m_ = shape[0];
    const long int n_ = shape[1];
    const long int k_ = shape[2];

    float alpha = 1.f, beta = 0.f;
    aligned_vector<uint8_t> Aint8(m * k);

    aligned_vector<int8_t> Bint8(k * n);

    aligned_vector<float> Cfp32_mkl(m * n);
    //aligned_vector<int32_t> Cint32_mkl(Cfp32_mkl.size());
    aligned_vector<float> Cfp32_mkldnn(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_mkl(Cfp32_mkl.size());    // XXX Why do we need this?
    aligned_vector<int32_t> Cint32_mkldnn(Cfp32_mkl.size()); // XXX Why do we need this?
    aligned_vector<int32_t> Cint32_mkl_s8u8s32(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_mkl_packapi_s8u8s32(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_mkldnn_s8u8s32(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_mkldnn_packapi_s8u8s32(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_ref(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb_acc32(Cfp32_mkl.size());
    aligned_vector<int32_t> Cint32_fb_acc16(Cfp32_mkl.size());

    // A matrix
    randFill<uint8_t>(Aint8, 0, 5);
    aligned_vector<float> Afp32(Aint8.begin(), Aint8.end());

    randFill<int8_t>(Bint8, -4, 4);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

    aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());

    double nops = 2.0 * m * n * k;
    double ttot = 0.0;
    string runType;

    matmul_u8i8acc32_ref(
        m, n, k, k, n, n, Aint8.data(), Bint8.data(), Cint32_ref.data());

#ifdef USE_MKL
    runType = "MKL_fp32";
    ttot = measureWithWarmup(
        [&]() {
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              m,
              n,
              k,
              alpha,
              Afp32.data(),
              k,
              Bfp32.data(),
              n,
              beta,
              Cfp32_mkl.data(),
              n);
        },
        NWARMUP,
        NITER,
        flush ? &llc : nullptr);
    ttot *= 1e9; // convert to ns

    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", "
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
         << setw(16) << 0 << ", " << setw(16) << 0 << ", " << setw(16) << 0
         << ", " << setw(16) << 0 << ", "
#endif
         << setw(5) << fixed << setw(5) << setprecision(1) << nops / ttot
         << endl;

    for (auto i = 0; i < Cfp32_mkl.size(); ++i) {
      Cint32_mkl[i] = (int32_t)Cfp32_mkl[i];
    }

    ttot = 0.0;
    runType = "MKL_s8u8";
    int32_t co[] = {0};
    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();
      cblas_gemm_s8u8s32(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans,
          CblasFixOffset,
          m,
          n,
          k,
          alpha,
          Aint8.data(),
          k,
          0, // A offset
          Bint8.data(),
          n,
          0, // B offset
          beta,
          Cint32_mkl_s8u8s32.data(),
          n,
          co);
      end = chrono::high_resolution_clock::now();
      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
      }
    }
    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", "
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
         << setw(16) << 0 << ", " << setw(16) << 0 << ", " << setw(16) << 0
         << ", " << setw(16) << 0 << ", "
#endif
         << setw(5) << fixed << setw(5) << setprecision(1) << (static_cast<double>(NITER) * nops) / ttot
         << endl;


    //vector<int32_t> row_offsets(m);
    compare_buffers(Cint32_ref.data(), Cint32_mkl_s8u8s32.data(), m, n, n, 5);

    //matmul_u8i8acc32_ref(
    //    m, n, k, k, n, n, Aint8.data(), Bint8.data(), Cint32_ref.data());
    ttot = 0.0;
    runType = "MKL_packapi_s8u8";

    size_t packedBSize = cblas_gemm_s8u8s32_pack_get_size(CblasBMatrix,
            m, n, k);
    void *packedB = (void *) mkl_malloc(packedBSize, 64);
    cblas_gemm_s8u8s32_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, m, n, k,
            Bint8.data(), n, packedB);
    // TODO Error handling.

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();
      cblas_gemm_s8u8s32_compute(
          CblasRowMajor,
          CblasNoTrans,
          CblasPacked,
          CblasFixOffset,
          m,
          n,
          k,
          alpha,
          Aint8.data(),
          k,
          0, // A offset
          packedB,
          n,
          0, // B offset
          beta,
          Cint32_mkl_packapi_s8u8s32.data(),
          n,
          co);
      end = chrono::high_resolution_clock::now();
      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
      }
    }
    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", "
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
         << setw(16) << 0 << ", " << setw(16) << 0 << ", " << setw(16) << 0
         << ", " << setw(16) << 0 << ", "
#endif
         << setw(5) << fixed << setw(5) << setprecision(1) << (static_cast<double>(NITER) * nops) / ttot
         << endl;

    mkl_free(packedB);
    compare_buffers(Cint32_ref.data(), Cint32_mkl_packapi_s8u8s32.data(), m, n, n, 5);
#endif

#ifdef USE_MKLDNN
    ttot = 0.0;
    runType = "DNN_fp32";
    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();
      mkldnn_sgemm(
          'N',
          'N',
          m_,
          n_,
          k_,
          1.0f, //alpha,
          Afp32.data(),
          k_,
          Bfp32.data(),
          n_,
          0.f, //beta,
          Cfp32_mkldnn.data(),
          n_);
      end = chrono::high_resolution_clock::now();
      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
      }
    }
    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", "
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
         << setw(16) << 0 << ", " << setw(16) << 0 << ", " << setw(16) << 0
         << ", " << setw(16) << 0 << ", "
#endif
         << setw(5) << fixed << setw(5) << setprecision(1) << (static_cast<double>(NITER) * nops) / ttot
         << endl;

    // XXX What is this for?
    for (auto i = 0; i < Cfp32_mkl.size(); ++i) {
      Cint32_mkldnn[i] = (int32_t)Cfp32_mkldnn[i];
    }

    ttot = 0.0;
    runType = "DNN_s8u8";
    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();
      mkldnn_gemm_u8s8s32(
          'N',
          'N',
          'F',
          m_,
          n_,
          k_,
          1.0f, //alpha,
          Aint8.data(),
          k_,
          0, // A offset
          Bint8.data(),
          n_,
          0, // B offset
          0.f, //beta,
          Cint32_mkldnn_s8u8s32.data(),
          n_,
          co);
      end = chrono::high_resolution_clock::now();
      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
      }
    }
    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", "
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
         << setw(16) << 0 << ", " << setw(16) << 0 << ", " << setw(16) << 0
         << ", " << setw(16) << 0 << ", "
#endif
         << setw(5) << fixed << setw(5) << setprecision(1) << (static_cast<double>(NITER) * nops) / ttot
         << endl;

    compare_buffers(Cint32_ref.data(), Cint32_mkldnn_s8u8s32.data(), m, n, n, 5);


    ttot = 0.0;
    runType = "DNN_packapi_s8u8";

    mkldnn::impl::cpu::gemm_s8u8s32_pack_get_size(
            "A", "N", "N", &n, &m, &k, &n, &k, &packedBSize);
    packedB = (void *) mkl_malloc(packedBSize, 64);
    mkldnn::impl::cpu::gemm_s8u8s32_pack("A", "N", "N", &n, &m, &k, &n, &k,
            Bint8.data(), packedB);
    // TODO Error handling.

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();
      mkldnn::impl::cpu::gemm_s8u8s32_compute(
          "P",
          "N",
          "F",
          &n,
          &m,
          &k,
          (const int8_t *)packedB,
          &n,
          (const uint8_t *)Aint8.data(),
          &k,
          &beta,
          Cint32_mkldnn_packapi_s8u8s32.data(),
          &n,
          co);
      end = chrono::high_resolution_clock::now();
      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
        ttot += dur.count();
      }
    }
    ((volatile char*)(llc.data()));

    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType << ", "
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
         << setw(16) << 0 << ", " << setw(16) << 0 << ", " << setw(16) << 0
         << ", " << setw(16) << 0 << ", "
#endif
         << setw(5) << fixed << setw(5) << setprecision(1) << (static_cast<double>(NITER) * nops) / ttot
         << endl;

    mkl_free(packedB);
    compare_buffers(Cint32_ref.data(), Cint32_mkldnn_packapi_s8u8s32.data(),
            m, n, n, 5);
#endif

    vector<int32_t> row_offsets(m);

    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Cint32_ref.data(),
    // m, n, n, "C int32");

    PackBMatrix<int8_t> packedB_int32(
        matrix_op_t::NoTranspose, k, n, Bint8.data(), n, nullptr, 1);

    ttot = 0.0;
    runType = "FBGEMM_i8_acc32";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    double total_packing_time = 0.0;
    double total_computing_time = 0.0;
    double total_kernel_time = 0.0;
    double total_postprocessing_time = 0.0;
    double total_run_time = 0.0;
#endif
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType;

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      packing_time = 0.0;
      computing_time = 0.0;
      kernel_time = 0.0;
      postprocessing_time = 0.0;
      run_time = 0.0;
#endif
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        PackAMatrix<uint8_t> packA_int32(
            matrix_op_t::NoTranspose, m, k, Aint8.data(), k, nullptr, 1);

        DoNothing<int32_t, int32_t> doNothing32BitObj;
        memCopy<> memcopyObj(doNothing32BitObj);
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();
        // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
        fbgemmPacked(
            packA_int32,
            packedB_int32,
            Cint32_fb_acc32.data(),
            Cint32_fb_acc32.data(),
            n,
            memcopyObj,
            tid,
            num_threads);
      }

      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
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
    ((volatile char*)(llc.data()));
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_fb.data(), m, n, n, "C fb");

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << ", " << setw(16) << total_packing_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_kernel_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_postprocessing_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_run_time / (double)NITER / 1e3;
#endif
    cout << ", " << setw(5) << fixed << setw(5) << setprecision(1)
         << NITER * nops / ttot << endl;

    compare_buffers(Cint32_ref.data(), Cint32_fb_acc32.data(), m, n, n, 5);

    PackBMatrix<int8_t, int16_t> packedB_int16(
        matrix_op_t::NoTranspose, k, n, Bint8.data(), n, nullptr, 1);

    ttot = 0.0;
    runType = "FBGEMM_i8_acc16";
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    total_packing_time = 0.0;
    total_computing_time = 0.0;
    total_kernel_time = 0.0;
    total_postprocessing_time = 0.0;
    total_run_time = 0.0;
#endif
    cout << setw(6) << m << ", " << setw(6) << n << ", " << setw(6) << k << ", "
         << setw(16) << runType;

    for (auto i = 0; i < NWARMUP + NITER; ++i) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      packing_time = 0.0;
      computing_time = 0.0;
      kernel_time = 0.0;
      postprocessing_time = 0.0;
      run_time = 0.0;
#endif
      llc_flush(llc);
      start = chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        PackAMatrix<uint8_t, int16_t> packA_int16(
            matrix_op_t::NoTranspose, m, k, Aint8.data(), k, nullptr, 1);

        DoNothing<int32_t, int32_t> doNothing32BitObj;
        memCopy<> memcopyObj(doNothing32BitObj);
        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();
        // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
        fbgemmPacked(
            packA_int16,
            packedB_int16,
            Cint32_fb_acc16.data(),
            Cint32_fb_acc16.data(),
            n,
            memcopyObj,
            tid,
            num_threads);
      }

      end = chrono::high_resolution_clock::now();

      if (i >= NWARMUP) {
        auto dur = chrono::duration_cast<chrono::nanoseconds>(end - start);
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
    ((volatile char*)(llc.data()));
    // printMatrix(matrix_op_t::NoTranspose, Bint8.data(), k, n, n, "B
    // unpacked");
    // printMatrix(matrix_op_t::NoTranspose, Aint8.data(), m, k, k,
    // "A unpacked");
    // printMatrix(matrix_op_t::NoTranspose,
    // Cint8_fb.data(), m, n, n, "C fb");
    // compare_buffers(row_offsets.data(), row_offset_buf.data(),
    // row_offsets.size(), 5);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    cout << ", " << setw(16) << total_packing_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_kernel_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_postprocessing_time / (double)NITER / 1e3 << ", "
         << setw(16) << total_run_time / (double)NITER / 1e3;
#endif
    cout << ", " << setw(5) << fixed << setw(5) << setprecision(1)
         << NITER * nops / ttot << endl;
    cout << endl;

    compare_buffers(Cint32_ref.data(), Cint32_fb_acc16.data(), m, n, n, 5);
  }
}

int main(int /* unused */, char** /* unused */) {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  performance_test();
  return 0;
}
