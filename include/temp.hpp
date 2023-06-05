#pragma once
#include "../include/file_buffer.hpp"
#include "../include/globals.hpp"
#include "../include/hk_seq.hpp"
#include "../include/parameters.hpp"

namespace epic
{

  int test_seq(epic::Parameters &parameters)
  {

    epic::FileBuffer file_buffer(parameters.fileQueries);
    if (file_buffer.read_file_to_buffer())
    {
      fprintf(stderr, "File was not read to buffer.");
      return 0;
    }
    fprintf(stderr, "File read in %f ms.\n", file_buffer.millis);

    auto start = START_TIME;

    epic::EpicSeq seq(file_buffer);
    if (seq.create())
      return 1;
    auto stop = STOP_TIME;
    float millis = DURATION_IN_MILLISECONDS(start, stop);
    fprintf(stderr, "Created kmerstring array in %f ms.\n", millis);
    fprintf(stderr, "\n\nNumber of letters: %" PRIu64 ".\n", (u64)seq.absolute_letter_count);

    fprintf(stderr, "\n\nNumber of words in kmerstring vector (size()): %" PRIu64 ".\n", (u64)seq.kmer_string.size());

    for (int i = 0; i < 1000; i++)
    {
      fprintf(stderr, "%" PRIx64 " ", seq.kmer_string[i]);
    }

    return 0;
  }
}