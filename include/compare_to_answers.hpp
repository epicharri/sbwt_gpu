#pragma once
#include "../include/globals.hpp"
#include "../include/gpu/device_stream.hpp"
#include "../include/kmer_string.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace epic
{

  struct CompareToAnswers
  {
    u64 number_of_errors = 0ULL;
    u64 number_of_minus_ones = 0ULL;
    std::vector<std::string> read_answer_lines(std::string);
    int check(KmerString &, std::string);
    CompareToAnswers() = default;
  };

  std::vector<std::string> CompareToAnswers::read_answer_lines(std::string filename)
  {
    std::vector<std::string> textlines;
    std::string textline;
    std::ifstream textfile(filename);
    if (!textfile.is_open())
    {
      fprintf(stderr, "Text file %s could not be opened.", filename);
      return textlines;
    }
    while (std::getline(textfile, textline))
    {
      textlines.push_back(textline);
    }
    textfile.close();
    return textlines;
  }

  int CompareToAnswers::check(KmerString &kmer_string, std::string filename_of_answers)
  {

    number_of_minus_ones = 0ULL;
    std::vector<uint32_t> answers_per_line;
    u32 answers_per_one_line = 1U;
    for (u32 i = 1U; i < kmer_string.number_of_kmer_positions; i += 1U)
    {
      if ((kmer_string.kmer_positions[i - 1U] + 1U) == kmer_string.kmer_positions[i])
      {
        answers_per_one_line += 1U;
        if (i == kmer_string.number_of_kmer_positions - 1U)
          answers_per_line.push_back(answers_per_one_line);
      }
      else
      {
        answers_per_line.push_back(answers_per_one_line);
        answers_per_one_line = 1U;
      }
    }
    number_of_errors = 0ULL;
    std::vector<std::string> answer_lines = read_answer_lines(filename_of_answers);
    if (answer_lines.size() == 0)
    {
      fprintf(stderr, "The file containing answers is empty.\n");
      fprintf(stderr, "CAN NOT FIND IF THERE ARE ERRORS OR NOT!\n");
      return 0;
    }

    if (answer_lines.size() != answers_per_line.size())
    {
      fprintf(stderr, "\nanswer_lines.size() = %" PRIu32 " but in GPU results there are %" PRIu32 " lines \n", (answer_lines.size()), (answers_per_line.size()));
      return 0;
    }
    u32 i_search_results = 0U;
    for (u32 i = 0U; i < answer_lines.size(); i += 1U)
    {
      std::string answer_line = answer_lines[i];
      std::string gpu_answer_line = "";
      for (u32 j = 0U; j < answers_per_line[i]; j += 1U)
      {
        gpu_answer_line += std::to_string((int64_t)kmer_string.search_results[i_search_results]) + " ";
        if (kmer_string.search_results[i_search_results] == 0xffffffffffffffffULL)
          number_of_minus_ones += 1ULL;
        i_search_results += 1U;
      }
      if (gpu_answer_line.compare(answer_line))
        number_of_errors += 1ULL;
    }
    if (number_of_errors)
    {
      fprintf(stderr, "\nERRORS: Results given by GPU search have errors in %" PRIu64 " lines of the read.\n\n", number_of_errors);
    }
    else
    {
      fprintf(stderr, "\nSUCCESS\n\n");
    }
    DEBUG_CODE(fprintf(stderr, "Number of minus ones is %" PRIu64 "\n", number_of_minus_ones);)
    return 0;
  }

}