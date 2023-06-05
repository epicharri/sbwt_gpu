#pragma once
#ifdef __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include "../enums.hpp"

typedef uint64_t u64;
typedef uint32_t u32;

#define LANE_2X_SB4096 ((threadIdx.x & 0b111) << 1)
#define LANE_2X ((threadIdx.x & 0b11) << 1)
// #define LANE_2X_SB1024 ((threadIdx.x & 0b11) << 1)
#define LANE_2X_SB1024 ((threadIdx.x & 0b1) << 1)

// A helper macro to make code portable between NVIDIA and AMD:
#ifdef __HIP_PLATFORM_HCC__
// AMD does not support the _sync variants
#define SHFL_XOR(val, offset) __shfl_xor(val, offset)
#else
// NVIDIA has deprecated the variant without _sync
#define SHFL_XOR(val, offset) __shfl_xor_sync(0xffffffff, val, offset)
// syncwarp() is needed for newer NVIDIA hardware (Volta V100 or newer)
#endif

namespace epic
{

    namespace gpu
    {

        /**
         * @brief
         *
         * @tparam superblock_size 256, 512, 1024, 2048, or 4096
         * @return __device__ int log_2 of the superblock size.
         */

        template <int superblock_size>
        __device__ constexpr int log_of_superblock_size()
        {
            if (superblock_size == 256)
                return 8U;
            if (superblock_size == 512)
                return 9U;
            if (superblock_size == 1024)
                return 10U;
            if (superblock_size == 2048)
                return 11U;
            if (superblock_size == 4096)
                return 12U;
        }

        template <int superblock_size>
        __device__ constexpr int log_of_basicblock_size()
        {
            if (superblock_size == 256)
                return 6U;
            if (superblock_size == 512)
                return 7U;
            if (superblock_size == 1024)
                return 8U;
            if (superblock_size == 2048)
                return 9U;
            if (superblock_size == 4096)
                return 10U;
        }

        template <int superblock_size>
        __device__ constexpr int superblock_size_minus_one()
        {
            return superblock_size - 1;
        }

        template <int superblock_size>
        __device__ constexpr u64 mask_of_vector_index()
        {
            if (superblock_size == 512)
                return ~(0ULL);
            if (superblock_size == 1024)
                return ~(1ULL);
            if (superblock_size == 2048)
                return ~(2ULL);
            if (superblock_size == 4096)
                return ~(3ULL);
        }

        template <int superblock_size>
        __device__ constexpr u32 mask_for_index_of_word_inside_basic_block()
        {
            if (superblock_size == 512)
                return 0b1U;
            if (superblock_size == 1024)
                return 0b11U;
            if (superblock_size == 2048)
                return 0b111U;
            if (superblock_size == 4096)
                return 0b1111U;
        }

        template <int superblock_size, typename T = ulonglong2, int offset_index_of_128_bit_vector>
        __forceinline__ __device__ T load_2_uint64_words(const u64 *__restrict__ data, const u64 index)
        {
            if (superblock_size == 512)
            {
                return reinterpret_cast<const T *>(data)[((index >> 7) << 0) + offset_index_of_128_bit_vector];
            }
            if (superblock_size == 1024)
            {
                return reinterpret_cast<const T *>(data)[((index >> 8) << 1) + offset_index_of_128_bit_vector];
            }
            if (superblock_size == 2048)
            {
                return reinterpret_cast<const T *>(data)[((index >> 9) << 2) + offset_index_of_128_bit_vector];
            }
            if (superblock_size == 4096)
            {
                return reinterpret_cast<const T *>(data)[((index >> 10) << 3) + offset_index_of_128_bit_vector];
            }
        }

        template <int superblock_size, typename T = ulonglong2>
        __forceinline__ __device__ T load_2_uint64_words_with_offset(const u64 *__restrict__ data, const u64 index, u32 offset_index_of_128_bit_vector)
        {
            if (superblock_size == 512)
            {
                return reinterpret_cast<const T *>(data)[((index >> 7) << 0) + offset_index_of_128_bit_vector];
            }
            if (superblock_size == 1024)
            {
                return reinterpret_cast<const T *>(data)[((index >> 8) << 1) + offset_index_of_128_bit_vector];
            }
            if (superblock_size == 2048)
            {
                return reinterpret_cast<const T *>(data)[((index >> 9) << 2) + offset_index_of_128_bit_vector];
            }
            if (superblock_size == 4096)
            {
                return reinterpret_cast<const T *>(data)[((index >> 10) << 3) + offset_index_of_128_bit_vector];
            }
        }

        template <int superblock_size>
        __device__ constexpr int size_of_data128_array()
        {
            if (superblock_size == 256)
                return 1;
            return superblock_size / 512;
        }

        template <int superblock_size, typename T = ulonglong2>
        __device__ void load_data_to_array(T *data128, const u64 *__restrict__ data, const u64 index)
        {
            if (superblock_size >= 512)
            {
                data128[0] = load_2_uint64_words<superblock_size, T, 0>(data, index);
            }
            if (superblock_size >= 1024)
            {
                data128[1] = load_2_uint64_words<superblock_size, T, 1>(data, index);
            }
            if (superblock_size >= 2048)
            {
                data128[2] = load_2_uint64_words<superblock_size, T, 2>(data, index);
                data128[3] = load_2_uint64_words<superblock_size, T, 3>(data, index);
            }
            if (superblock_size >= 4096)
            {
                data128[4] = load_2_uint64_words<superblock_size, T, 4>(data, index);
                data128[5] = load_2_uint64_words<superblock_size, T, 5>(data, index);
                data128[6] = load_2_uint64_words<superblock_size, T, 6>(data, index);
                data128[7] = load_2_uint64_words<superblock_size, T, 7>(data, index);
            }
        }

        template <int superblock_size>
        __forceinline__ __device__ u64 popcount_inside_basic_block_normal_with_loop(const u64 *__restrict__ data, const u64 i)
        {
            u64 popcount = 0ULL;
            if (superblock_size == 256)
            {
                popcount = __popcll(data[i >> 6] >> (64U - (i & 63U)));
            }
            else
            {
                ulonglong2 data128[size_of_data128_array<superblock_size>()];
                load_data_to_array<superblock_size>(data128, data, i);

                u32 i_target_word = (i >> 6) & mask_for_index_of_word_inside_basic_block<superblock_size>();
                u64 mask = 0xffff'ffff'ffff'ffffULL;
                u64 target_word_mask = mask << (64U - (i & 63U));
#pragma unroll
                for (u32 i_word = 0U; i_word < (size_of_data128_array<superblock_size>() * 2U); i_word += 2U)
                {
                    if (i_word == i_target_word)
                        mask = target_word_mask;
                    if (i_word > i_target_word)
                        mask = 0ULL;
                    popcount += __popcll(data128[i_word / 2].x & mask);
                    if ((i_word + 1U) == i_target_word)
                        mask = target_word_mask;
                    if ((i_word + 1U) > i_target_word)
                        mask = 0ULL;
                    popcount += __popcll(data128[i_word / 2].y & mask);
                }
            }
            return popcount;
        }

        template <int superblock_size>
        __forceinline__ __device__ u64 popcount_inside_basic_block_normal(const u64 *__restrict__ data, const u64 i)
        {
            u64 popcount;
            if (superblock_size == 256)
            {
                popcount = __popcll(data[i >> 6] >> (64U - (i & 63U)));
            }
            else
            {
                ulonglong2 data128[size_of_data128_array<superblock_size>()];
                load_data_to_array<superblock_size, ulonglong2>(data128, data, i);

                u32 iW = (i >> 6) & mask_for_index_of_word_inside_basic_block<superblock_size>();
                u32 targetSHR = 64U - (i & 63U);
                popcount =
                    __popcll(data128[0].x >> ((0U == iW) * targetSHR + 0U)) +
                    __popcll(data128[0].y >> ((1U == iW) * targetSHR + ((1U > iW) << 6)));
                if (superblock_size >= 1024)
                {
                    popcount +=
                        __popcll(data128[1].x >> ((2U == iW) * targetSHR + ((2U > iW) << 6))) +
                        __popcll(data128[1].y >> ((3U == iW) * targetSHR + ((3U > iW) << 6)));
                }
                if (superblock_size >= 2048)
                {
                    popcount +=
                        __popcll(data128[2].x >> ((4U == iW) * targetSHR + ((4U > iW) << 6))) +
                        __popcll(data128[2].y >> ((5U == iW) * targetSHR + ((5U > iW) << 6))) +
                        __popcll(data128[3].x >> ((6U == iW) * targetSHR + ((6U > iW) << 6))) +
                        __popcll(data128[3].y >> ((7U == iW) * targetSHR + ((7U > iW) << 6)));
                }
                if (superblock_size == 4096)
                {
                    popcount +=
                        __popcll(data128[4].x >> ((8U == iW) * targetSHR + ((8U > iW) << 6))) + __popcll(data128[4].y >> ((9U == iW) * targetSHR + ((9U > iW) << 6))) + __popcll(data128[5].x >> ((10U == iW) * targetSHR + ((10U > iW) << 6))) + __popcll(data128[5].y >> ((11U == iW) * targetSHR + ((11U > iW) << 6))) + __popcll(data128[6].x >> ((12U == iW) * targetSHR + ((12U > iW) << 6))) + __popcll(data128[6].y >> ((13U == iW) * targetSHR + ((13U > iW) << 6))) + __popcll(data128[7].x >> ((14U == iW) * targetSHR + ((14U > iW) << 6))) + __popcll(data128[7].y >> ((15U == iW) * targetSHR + ((15U > iW) << 6)));
                }
            }
            return popcount;
        }

        template <int superblock_size, int rank_version = epic::kind::poppy>
        __forceinline__ __device__ u64 popcount_before_basic_block(const u64 *__restrict__ data, const u64 *__restrict__ L0, u64 *L12, const u64 i)
        {
            u64 popcount;

            if (rank_version == epic::kind::poppy)
            {
                if (superblock_size == 4096)
                {
                    u64 entryL12 = L12[i >> log_of_superblock_size<superblock_size>()];

                    u64 entryL12BlockCounts = (entryL12 & 0x1ffffffffULL) >> ((3U - ((i & 4095U) >> 10)) * 11U);
                    popcount = L0[i >> 31] + (entryL12 >> 33) + (((entryL12BlockCounts & 0x7ffU) + ((entryL12BlockCounts >> 11) & 0x7ffU) + (entryL12BlockCounts >> 22)));
                }

                if (superblock_size < 4096)
                {
                    u64 entryL12 = L12[i >> log_of_superblock_size<superblock_size>()];

                    u32 entryL12BlockCounts = (entryL12 & 0x3fff'ffffU) >>
                                              ((3U - ((i & superblock_size_minus_one<superblock_size>()) >> log_of_basicblock_size<superblock_size>())) * 10U);
                    popcount = L0[i >> 32] + (entryL12 >> 32) + (((entryL12BlockCounts & 0x3ffU) + ((entryL12BlockCounts >> 10) & 0x3ffU) + ((entryL12BlockCounts >> 20) & 0x3ffU)));
                }
            }

            if (rank_version == epic::kind::cum_poppy)
            {
                u64 entryL12 = L12[i >> log_of_superblock_size<superblock_size>()];

                if (superblock_size == 2048)
                {
                    u32 entryL12BlockCounts = (entryL12 & 0xffff'ffffU) >>
                                              ((3U - ((i & superblock_size_minus_one<superblock_size>()) >> log_of_basicblock_size<superblock_size>())) * 11U);
                    popcount = L0[i >> 32] + (entryL12 >> 32) + (entryL12BlockCounts & 0x7ffU);
                }
                if (superblock_size < 2048)
                {
                    u32 entryL12BlockCounts = (entryL12 & 0x3fff'ffffU) >>
                                              ((3U - ((i & superblock_size_minus_one<superblock_size>()) >> log_of_basicblock_size<superblock_size>())) * 10U);
                    popcount = L0[i >> 32] + (entryL12 >> 32) + (entryL12BlockCounts & 0x3ffU);
                }
            }
            return popcount;
        }

        __forceinline__ __device__ u64 popcount_inside_basic_block_shuffles_2048(const u64 *__restrict__ data, const u64 i)
        {
            u64 m;
            u64 mask;
            ulonglong2 data128;
            u64 targetWordMask;
            u32 kLocalIndex;
            u32 popc;
            u32 mypopc;
            u32 iW;

#pragma unroll(4)
            for (u32 lane = 0U; lane < 8U; lane += 2U)
            {
                kLocalIndex = 0U;
                m = 0ULL;
                if (LANE_2X == lane)
                { // 0, 2, 4, 6
                    kLocalIndex = (u32)(i & 511ULL);
                    m = (u64)(&reinterpret_cast<const ulonglong2 *>(data)[((i >> 9) << 2)]);
                }
                kLocalIndex |= SHFL_XOR(kLocalIndex, 1);
                m |= SHFL_XOR(m, 1);
                kLocalIndex |= SHFL_XOR(kLocalIndex, 2);
                m |= SHFL_XOR(m, 2);
                data128 = *((const ulonglong2 *)(m + (8U * LANE_2X)));
                targetWordMask = 0xffffffffffffffffULL << (64U - (kLocalIndex & 63U));
                iW = (kLocalIndex >> 6) & 0b111U;
                mask = 0xffffffffffffffffULL;
                if (LANE_2X == iW)
                    mask = targetWordMask;
                if (LANE_2X > iW)
                    mask = 0ULL;
                popc = __popcll(data128.x & mask);
                mask = 0xffffffffffffffffULL;
                if ((LANE_2X + 1) == iW)
                    mask = targetWordMask;
                if ((LANE_2X + 1) > iW)
                    mask = 0ULL;
                popc += __popcll(data128.y & mask);
                popc += SHFL_XOR(popc, 1);
                popc += SHFL_XOR(popc, 2);
                if (LANE_2X == lane)
                { // 0,2,4,6
                    mypopc = popc;
                }
            }
            return mypopc;
        }

        __forceinline__ __device__ u64 popcount_inside_basic_block_shuffles_1024(const u64 *__restrict__ data, const u64 i)
        {
            u64 m;
            u64 mask;
            ulonglong2 data128;
            u64 targetWordMask;
            u32 kLocalIndex;
            u32 popc;
            u32 mypopc;
            u32 iW;

#pragma unroll(2)
            for (u32 lane = 0U; lane < 4U; lane += 2U)
            {
                kLocalIndex = 0U;
                m = 0ULL;
                if (LANE_2X_SB1024 == lane)
                { // 0, 2
                    kLocalIndex = (u32)(i & 255ULL);
                    m = (u64)(&reinterpret_cast<const ulonglong2 *>(data)[((i >> 8) << 1)]);
                }
                kLocalIndex |= SHFL_XOR(kLocalIndex, 1);
                m |= SHFL_XOR(m, 1);

                data128 = *((const ulonglong2 *)(m + (8U * LANE_2X_SB1024)));

                targetWordMask = 0xffffffffffffffffULL << (64U - (kLocalIndex & 63U));
                iW = (kLocalIndex >> 6) & 0b11U;
                mask = 0xffffffffffffffffULL;
                if (LANE_2X_SB1024 == iW)
                    mask = targetWordMask;
                if (LANE_2X_SB1024 > iW)
                    mask = 0ULL;
                popc = __popcll(data128.x & mask);
                mask = 0xffffffffffffffffULL;
                if ((LANE_2X_SB1024 + 1) == iW)
                    mask = targetWordMask;
                if ((LANE_2X_SB1024 + 1) > iW)
                    mask = 0ULL;
                popc += __popcll(data128.y & mask);
                popc += SHFL_XOR(popc, 1);

                if (LANE_2X_SB1024 == lane)
                { // 0, 2
                    mypopc = popc;
                }
            }
            return mypopc;
        }

        template <int superblock_size, bool shuffles = 0>
        __forceinline__ __device__ u64 popcount_inside_basic_block(const u64 *__restrict__ data, const u64 i)
        {
            if (superblock_size == 1024 && shuffles == 1)
                return popcount_inside_basic_block_shuffles_1024(data, i);
            if (superblock_size == 2048 && shuffles == 1)
                return popcount_inside_basic_block_shuffles_2048(data, i);
            else
                return popcount_inside_basic_block_normal<superblock_size>(data, i);
        }

        template <int superblock_size, bool shuffles = false, int rank_version = epic::kind::poppy>
        __forceinline__ __device__ u64 rank(const u64 *__restrict__ data, const u64 *__restrict__ L0, u64 *L12, const u64 i)
        {
            u64 r = popcount_inside_basic_block<superblock_size, shuffles>(data, i);
            r += popcount_before_basic_block<superblock_size, rank_version>(data, L0, L12, i);
            return r;
        }
    }
}