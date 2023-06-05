#pragma once
#include "../include/bit_vector.hpp"
#include "../include/enums.hpp"
#include "../include/globals.hpp"
#include "../include/parameters.hpp"
#include <fstream>
#include <string>
#include <vector>

typedef uint64_t u64;
typedef uint32_t u32;

namespace epic
{
  class RankDataStructures
  {
  public:
    u32 bits_in_superblock = 0U;        // Number of bits in a superblock.
    u32 bits_in_basicblock = 0U;        // Number of bits in a basic block.
    u32 log_2_of_hyperblock_size = 32U; // Number of bits in a hyperblock is 2 powered to log_2_of_hyperblock_size.
    int rank_version = epic::kind::poppy;
    u64 number_of_words_padded_layer_0 = 0ULL;
    u64 number_of_words_padded_layer_12 = 0ULL;
    u64 absolute_number_of_ones = 0ULL;
    u64 *layer_0 = nullptr;  // An array to store absolute cumulative number of 1s before each hyper block.
    u64 *layer_12 = nullptr; // An interleaved array to store relative cumulative number of 1s before each superblock and an absolute number of 1s inside a basic block.
    int construct(epic::BitVector &, u64);

    void create(const u32, const int, BitVector &);
    RankDataStructures() = default;
    ~RankDataStructures();

  private:
    inline u64 swap_bytes(u64 x);
    inline u64 round_up_first_to_multiple_of_second(u64, u64);
    int allocate_memory_for_the_structures();
    template <u32 words_in_basicblock>
    inline u32 popcount_basicblock(u64 *, u64);
    int precount_the_structures(epic::BitVector &, u64);
    template <u32 words_in_basicblock>
    inline int precount_the_structures_based_on_words_in_basicblock(epic::BitVector &, u64);
    template <u32 words_in_basicblock>
    inline int precount(epic::BitVector &);
  };

  RankDataStructures::~RankDataStructures()
  {
    DEBUG_BEFORE_DESTRUCT("RankDataStructures.layer_0");
    if (layer_0 != nullptr)
      delete[] layer_0;
    DEBUG_AFTER_DESTRUCT("RankDataStructures.layer_0");

    DEBUG_BEFORE_DESTRUCT("RankDataStructures.layer_12");
    if (layer_12 != nullptr)
      delete[] layer_12;
    DEBUG_AFTER_DESTRUCT("RankDataStructures.layer_12");
  };

  void RankDataStructures::create(const u32 t_bits_in_superblock, const int t_rank_version, epic::BitVector &bit_vector)
  {
    rank_version = t_rank_version;
    bits_in_superblock = t_bits_in_superblock;
    bits_in_basicblock = t_bits_in_superblock / 4U;
    log_2_of_hyperblock_size = 32U;
    if (t_bits_in_superblock == 4096U)
      log_2_of_hyperblock_size = 31U;
    number_of_words_padded_layer_0 = 1ULL + (bit_vector.number_of_bits >> log_2_of_hyperblock_size);
    number_of_words_padded_layer_12 = 1ULL + (round_up_first_to_multiple_of_second(bit_vector.number_of_bits, (u64)bits_in_superblock)) / (u64)bits_in_superblock;
  }

  int RankDataStructures::allocate_memory_for_the_structures()
  {
    try
    {
      layer_0 = new u64[number_of_words_padded_layer_0];
      layer_12 = new u64[number_of_words_padded_layer_12];
      if (!(layer_0 && layer_12))
        return 1;
    }
    catch (const std::bad_alloc &e)
    {
      fprintf(stderr, "There is not enough memory for the rank data structures.\n");
      return 1;
    }
    return 0;
  };

  template <u32 words_in_basicblock>
  inline u32 RankDataStructures::popcount_basicblock(u64 *data, u64 i)
  {
    u32 popcount_of_basicblock = 0;
    u64 word;
#pragma unroll
    for (u32 j = 0U; j < words_in_basicblock; j += 1U)
    {
      word = data[i + j];
      popcount_of_basicblock += __builtin_popcountll(word);
      data[i + j] = swap_bytes(word); // This is needed to convert the endianess.
    }
    return popcount_of_basicblock;
  }

  inline u64 RankDataStructures::swap_bytes(u64 x)
  {
    return (x << 56) |
           ((x & 0x00'00'00'00'00'00'ff'00ULL) << 40) |
           ((x & 0x00'00'00'00'00'ff'00'00ULL) << 24) |
           ((x & 0x00'00'00'00'ff'00'00'00ULL) << 8) |
           ((x & 0x00'00'00'ff'00'00'00'00ULL) >> 8) |
           ((x & 0x00'00'ff'00'00'00'00'00ULL) >> 24) |
           ((x & 0x00'ff'00'00'00'00'00'00ULL) >> 40) |
           (x >> 56);
  };

  inline u64 RankDataStructures::round_up_first_to_multiple_of_second(u64 t_number, u64 t_coefficient)
  {
    return ((t_number + t_coefficient - 1ULL) / t_coefficient) * t_coefficient;
  }

  int RankDataStructures::precount_the_structures(epic::BitVector &bit_vector, u64 abs_count_before = 0ULL)
  {
    if (bits_in_superblock == 256)
      return precount_the_structures_based_on_words_in_basicblock<1>(bit_vector, abs_count_before);
    else if (bits_in_superblock == 512)
      return precount_the_structures_based_on_words_in_basicblock<2>(bit_vector, abs_count_before);
    else if (bits_in_superblock == 1024)
      return precount_the_structures_based_on_words_in_basicblock<4>(bit_vector, abs_count_before);
    else if (bits_in_superblock == 2048)
      return precount_the_structures_based_on_words_in_basicblock<8>(bit_vector, abs_count_before);
    else if (bits_in_superblock == 4096)
      return precount_the_structures_based_on_words_in_basicblock<16>(bit_vector, abs_count_before);
    else
      return 1;
  }

  template <u32 words_in_basicblock>
  inline int RankDataStructures::precount(epic::BitVector &bit_vector)
  {
    return 0;
  }

  template <u32 words_in_basicblock>
  inline int RankDataStructures::precount_the_structures_based_on_words_in_basicblock(epic::BitVector &bit_vector, u64 abs_count_before = 0ULL)
  {
    u64 bits_in_hyperblock = 1ULL << log_2_of_hyperblock_size;
    u64 words_in_hyperblock = (bits_in_hyperblock / ((u64)bits_in_superblock));
    u32 one_if_sb_4096 = (u32)(bits_in_superblock == 4096U);
    u64 i_data = 0ULL;
    u64 i_layer_12 = 0ULL;
    absolute_number_of_ones = abs_count_before;
    u32 rel_count = 0U;
    for (u32 i_layer_0 = 0U; i_layer_0 < number_of_words_padded_layer_0; i_layer_0++)
    {
      layer_0[i_layer_0] = absolute_number_of_ones;
      rel_count = 0U;
      for (u32 i_layer_12_rel = 0U; i_layer_12_rel < words_in_hyperblock; i_layer_12_rel += 1U)
      {
        if (i_layer_12 >= number_of_words_padded_layer_12 - 1ULL)
        {
          layer_12[i_layer_12] = ((u64)rel_count) << (32 + one_if_sb_4096);
          return 0;
        }
        u32 countBB[4];
#pragma unroll
        for (u32 b = 0U; b < 4U; b++)
        {
          countBB[b] = popcount_basicblock<words_in_basicblock>(bit_vector.data, i_data + (u64)(b * words_in_basicblock));
        }
        i_data += (u64)(words_in_basicblock << 2);
        if (rank_version == epic::kind::poppy)
        {
          layer_12[i_layer_12] = (((u64)rel_count) << (32 + one_if_sb_4096)) | ((((u64)countBB[0]) << (20 + 2 * one_if_sb_4096)) | (((u64)countBB[1]) << (10 + one_if_sb_4096)) | ((u64)countBB[2]));
        }
        else if (rank_version == epic::kind::cum_poppy)
        { // Only if superblock size is 256, 512, 1024, or 2048
          if (bits_in_superblock < 2048U)
          {
            layer_12[i_layer_12] = (((u64)rel_count) << 32) |
                                   ((((u64)countBB[0]) << 20) |
                                    (((u64)(countBB[0] + countBB[1])) << 10) |
                                    ((u64)(countBB[0] + countBB[1] + countBB[2])));
          }
          if (bits_in_superblock == 2048U)
          {
            layer_12[i_layer_12] = (((u64)rel_count) << 32) |
                                   ((((u64)countBB[0]) << 22) |
                                    (((u64)(countBB[0] + countBB[1])) << 11) |
                                    ((u64)(countBB[0] + countBB[1] + countBB[2])));
          }
        }
        i_layer_12 += 1ULL;
        u32 countSB = countBB[0] + countBB[1] + countBB[2] + countBB[3];
        absolute_number_of_ones += (u64)countSB;
        rel_count += countSB;
      }
    }
    return 0;
  }

  int RankDataStructures::construct(epic::BitVector &bit_vector, u64 abs_count_before = 0ULL)
  {
    if (!allocate_memory_for_the_structures())
    {
      return precount_the_structures(bit_vector, abs_count_before);
    }
    else
    {
      fprintf(stderr, "Unable to allocate memory for the rank data structures.\n");
      return 1;
    }
  }

}