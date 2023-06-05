#pragma once
#include "../include/globals.hpp"
#include "../include/gpu/device_stream.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace epic
{
  struct BitVector
  {
    u64 *data = nullptr;
    u64 number_of_bits = 0ULL;
    u64 number_of_bytes = 0ULL;
    u64 number_of_words = 0ULL;
    u64 number_of_words_padded = 0ULL;
    std::string filename = "";
    u64 round_up_first_to_multiple_of_second(u64 t_number, u64 t_coefficient);
    int allocate_memory_for_data_array();
    int create(std::string);

    bool system_is_little_endian();
    inline u64 swap_bytes(u64 x);
    int convert_endianess_of_bit_vector();
    int read_bit_vector_from_file();
    int read();

    BitVector() = default;
    ~BitVector();
  };

  BitVector::~BitVector()
  {
    DEBUG_BEFORE_DESTRUCT("BitVector.data");
    if (data)
    {
      delete[] data;
    }
    DEBUG_AFTER_DESTRUCT("BitVector.data");
  };

  int BitVector::allocate_memory_for_data_array()
  {
    try
    {
      data = new u64[number_of_words_padded];
      if (data)
        return 0;
    }
    catch (const std::bad_alloc &e)
    {
      fprintf(stderr, "There is not enough memory for a bit vector of %" PRIu64 " bits.\n", number_of_bits);
      return 1;
    }
    return 0;
  }

  inline u64 BitVector::round_up_first_to_multiple_of_second(u64 t_number, u64 t_coefficient)
  {
    return ((t_number + t_coefficient - 1ULL) / t_coefficient) * t_coefficient;
  }

  int BitVector::create(std::string t_filename)
  {
    filename = t_filename;
    std::ifstream file(filename, std::ios::in | std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
      fprintf(stderr, "The file %s can not be opened.\n", filename);
      return 1;
    }
    u64 size_of_file_in_bytes = file.tellg();
    if (size_of_file_in_bytes < 8)
    {
      fprintf(stderr, "The size of the file %s is too small to contain any data.\n", filename);
      return 1;
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer_for_size;
    buffer_for_size.reserve(8);
    file.read(&buffer_for_size[0], 8);
    u64 number_of_bits_in_file = 0ULL;
    for (int i = 0; i < 8; i++)
    {
      number_of_bits_in_file = number_of_bits_in_file | ((((uint64_t)buffer_for_size[i]) & 0xffULL) << (i * 8));
    }
    DEBUG_CODE(fprintf(stderr, "Number of bits in the file is %" PRIu64 ".\n", number_of_bits_in_file);)
    if (number_of_bits_in_file == 0ULL)
    {
      fprintf(stderr, "The first 8 bytes in the file %s indicates that there is a bit vector of size 0 in the file.\n", filename.c_str());
      return 1;
    }

    number_of_bytes = (number_of_bits_in_file + 7ULL) / 8ULL;
    if (size_of_file_in_bytes - 8ULL < number_of_bytes)
    {
      fprintf(stderr, "The first 8 bytes in the file %s indicates that the size of the bit vector is %" PRIu64 " bytes. However, there is only %" PRIu64 " bytes of data in the file.\n", filename, number_of_bytes, (size_of_file_in_bytes - 8ULL));
      return 1;
    }

    number_of_words = (round_up_first_to_multiple_of_second(number_of_bits_in_file, 64ULL)) / 64ULL;
    number_of_words_padded = (round_up_first_to_multiple_of_second(number_of_bits_in_file, 4096ULL) / 64ULL);
    number_of_bits = number_of_bits_in_file;
    file.close();

    return 0;
  }

  inline int BitVector::read_bit_vector_from_file()
  {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
      fprintf(stderr, "The file %s can not be opened.\n", filename);
      return 1;
    }
    for (u64 i = number_of_words; i < number_of_words_padded; i += 1ULL)
    {
      data[i] = 0ULL;
    }
    file.seekg(8, std::ios::beg);
    file.read(reinterpret_cast<char *>(data), number_of_bytes);
    // We should check that read is good.
    DEBUG_CODE(fprintf(stderr, "File %s read successfully.\n", filename.c_str());)
    return 0;
  }

  int BitVector::read()
  {
    if (allocate_memory_for_data_array())
      return 1;
    if (read_bit_vector_from_file())
      return 1;
    //   Converting endianess of bit vector() is done during construction of rank data structures.
    return 0;
  }

  bool BitVector::system_is_little_endian()
  {
    u64 number = 0x0807060504030201ULL;

    bool answer = true;
    u8 *bytes = reinterpret_cast<u8 *>(&number);
    for (u32 i = 0U; i < 8U; i++)
    {
      if (bytes[i] != (i + 1U))
        answer = false;
    }
    return answer;
  }

  inline u64 BitVector::swap_bytes(u64 x)
  {
    return (x << 56) | ((x & 0x00'00'00'00'00'00'ff'00ULL) << 40) | ((x & 0x00'00'00'00'00'ff'00'00ULL) << 24) | ((x & 0x00'00'00'00'ff'00'00'00ULL) << 8) | ((x & 0x00'00'00'ff'00'00'00'00ULL) >> 8) | ((x & 0x00'00'ff'00'00'00'00'00ULL) >> 24) | ((x & 0x00'ff'00'00'00'00'00'00ULL) >> 40) | (x >> 56);
  }

  int BitVector::convert_endianess_of_bit_vector()
  {
    if (system_is_little_endian())
    {
      for (u64 i = 0; i < number_of_words_padded; i += 1ULL)
      {
        data[i] = swap_bytes(data[i]);
      }
    }
    return 0;
  }
}