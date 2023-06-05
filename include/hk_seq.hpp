#pragma once
#include "../include/file_buffer.hpp"
#include "../include/globals.hpp"
#include "../include/utils/helpers.hpp"
#include <vector>

namespace epic
{
  class EpicSeq
  {
  public:
    epic::FileBuffer buffer;
    std::vector<char>::iterator iter;
    std::vector<u64> kmer_string;
    std::vector<u32> row_lengths;
    u64 i_letter = 0ULL; // Index of a character.
    u64 i_kmer_string = 0ULL;
    u64 absolute_letter_count = 0ULL;
    int create();
    EpicSeq(epic::FileBuffer &t_buffer);
    ~EpicSeq();

    int allocate_memory();
    void skip_header_line();
    void read_letter_line();
    inline void shift_last_word_if_needed();
    void shrink_size_of_kmer_string();

    u8 bits[256] = {};
  };

  EpicSeq::EpicSeq(epic::FileBuffer &t_buffer)
  {
    try
    {
      buffer = t_buffer;
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
    }
    i_letter = 0ULL;
    i_kmer_string = 0ULL;
    absolute_letter_count = 0ULL;

    bits['A'] = bits['a'] = 0U;
    bits['C'] = bits['c'] = 1U;
    bits['G'] = bits['g'] = 2U;
    bits['T'] = bits['t'] = 3U;
  }

  EpicSeq::~EpicSeq()
  {
  }

  int EpicSeq::allocate_memory()
  {
    try
    {
      kmer_string.reserve(buffer.size_of_file / 32); // 32 chars needs 64 bits.
    }
    catch (const std::bad_alloc &e)
    {
      std::cerr << e.what() << ": seq kmer string reserve " << (u64)(buffer.size_of_file / 32) << " u64 integers\n";
      return 1;
    }
    try
    {
      row_lengths.reserve(buffer.size_of_file / 100); // Enough, if the line size is 100 chars.
    }
    catch (const std::bad_alloc &e)
    {
      std::cerr << e.what() << ": seq row length reserve " << ((u64)buffer.size_of_file / 100) << " u64 integers.\n";
      return 1;
    }
    return 0;
  }

  void EpicSeq::skip_header_line()
  {
    while ((iter < buffer.data.end()) && (*iter != '\n'))
      iter++;
  }

  // Here it is assumed that the letter line consists of valid characters a,c,g,t,A,C,G, and T.
  //   In addition, it is assumed that each line, including the last line, ends with '\n'
  void EpicSeq::read_letter_line()
  {
    u64 word = 0ULL;
    u32 number_of_letters_in_row = 0U;
    while ((iter < buffer.data.end()) && (*iter != '\n'))
    {
      word = kmer_string[i_letter >> 5] << 2;
      kmer_string[i_letter >> 5] = word | ((u64)bits[*iter]);
      i_letter++;
      number_of_letters_in_row++;
      iter++;
    }
    row_lengths.push_back(number_of_letters_in_row);
    absolute_letter_count += number_of_letters_in_row;
  }

  inline void EpicSeq::shift_last_word_if_needed()
  {
    u64 shift_left_n_letters = absolute_letter_count & 0b11111ULL;
    if (shift_left_n_letters)
    {
      kmer_string[(absolute_letter_count - 1ULL) >> 5] <<= (2 * (32 - shift_left_n_letters));
    }
  }

  void EpicSeq::shrink_size_of_kmer_string()
  {
    u64 number_of_words_in_kmer_string = epic::utils::round_up_first_to_multiple_of_second<u64>(absolute_letter_count, 32ULL) / 32ULL + 1ULL;
    kmer_string.resize(number_of_words_in_kmer_string);
    kmer_string.shrink_to_fit();
  }

  int EpicSeq::create()
  {

    if (allocate_memory())
      return 1;

    try
    {
      iter = buffer.data.begin();
    }
    catch (const std::exception &e)
    {
      std::cerr << "iter = buffer.data.begin()" << e.what() << '\n';
      return 1;
    }

    for (; iter < buffer.data.end(); iter++)
    {
      switch (*iter)
      {
      case '>':
        skip_header_line();
        break;

      default:
        read_letter_line();
      }
    }
    shift_last_word_if_needed();
    shrink_size_of_kmer_string();
    row_lengths.shrink_to_fit();
    return 0;
  }
}