#pragma once
#include <iostream>

namespace epic
{
  namespace utils
  {
    template <typename T>
    inline T round_up_first_to_multiple_of_second(T t_number, T t_coefficient)
    {
      return ((t_number + t_coefficient - (T)1) / t_coefficient) * t_coefficient;
    }

    template <typename T>
    inline void print_host_error(T &e, const char *msg)
    {
      fprintf(stderr, "ERROR: %s: (%s)", e.what(), msg);
    }

    template <typename T>
    int allocate_host_memory(T *&host_pointer, std::size_t number_of_bytes, const char *msg = "")
    {
      try
      {
        host_pointer = new T[number_of_bytes];
      }
      catch (const std::bad_alloc &e)
      {
        print_host_error(e, msg);
        return 1;
      }
      return 0;
    }

  }
}