#ifndef _DATASET_H_
#define _DATASET_H_

#include <vector>
#include <fstream>
class Dataset
{

  public:

    Dataset() {}
    ~Dataset() 
    {
      query_batch_fasta.close();
      target_batch_fasta.close();
    }
    std::ifstream query_batch_fasta;
    std::ifstream target_batch_fasta;



    std::vector<std::string> query_seqs;
    std::vector<std::string> target_seqs;
    int total_seqs = 0;
    uint32_t maximum_sequence_length = 0;
    uint32_t target_seqs_len = 0;
    uint32_t query_seqs_len = 0;

    void read_input(std::string query_filename, std::string target_filename);

};

#endif
