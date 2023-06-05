#pragma once
#include "../include/globals.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace epic
{
  class FileBuffer
  {

  public:
    std::vector<char> data;
    std::string filename = "";
    std::size_t size_of_file = 0; // In bytes.
    int read_size_of_file();
    int read_file_to_buffer(std::size_t pos);
    float millis = 0.0;
    FileBuffer() = default;
    FileBuffer(std::string);
    ~FileBuffer();

  private:
    int allocate_memory_for_buffer();
  };

  FileBuffer::FileBuffer(std::string t_filename)
  {
    filename = t_filename;
  }

  FileBuffer::~FileBuffer()
  {
  }

  int FileBuffer::allocate_memory_for_buffer()
  {
    if (size_of_file == 0)
    {
      fprintf(stderr, "Size of the file is 0.\n");
      return 1;
    }
    try
    {
      data.reserve(size_of_file);
      data.resize(size_of_file);
    }
    catch (const std::bad_alloc &bad_alloc_error)
    {
      fprintf(stderr, "Size of the file: %zu \n", size_of_file);
      std::cerr << "ERROR: " << bad_alloc_error.what() << "\n";
      return 1;
    }
    return 0;
  }

  int FileBuffer::read_file_to_buffer(std::size_t pos = 0)
  {
    auto start = START_TIME;

    if (size_of_file == 0)
    {
      if (read_size_of_file())
        return 1;
    }

    if (allocate_memory_for_buffer())
      return 1;

    std::ifstream fs(filename, std::ios::in | std::ios::ate | std::ios::binary);

    if (!fs.is_open())
    {
      fprintf(stderr, "The file %s can not be opened.\n", filename.c_str());
      return 1;
    }

    fs.seekg(pos, std::ios::beg);
    fs.read(data.data(), size_of_file - pos);
    data.resize(size_of_file);
    data.shrink_to_fit();
    auto stop = STOP_TIME;
    millis = DURATION_IN_MILLISECONDS(start, stop);

    return 0;
  }

  int FileBuffer::read_size_of_file()
  {
    std::ifstream fs(filename, std::ios::in | std::ios::ate | std::ios::binary);
    if (!fs.is_open())
    {
      fprintf(stderr, "The file %s can not be opened.\n", filename.c_str());
      return 1;
    }
    size_of_file = fs.tellg();
    if (size_of_file == 0)
    {
      fprintf(stderr, "The size of the file %s is too small to contain any data.\n", filename.c_str());
      return 1;
    }
    fprintf(stderr, "Opening the file %s succeeded.", filename.c_str());
    fprintf(stderr, "Number of bytes in file: %zu \n", size_of_file);
    return 0;
  }

}