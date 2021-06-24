#include "dataset.h"
#include <iostream>
#include <string>
#include <string.h>

#define MAX(a,b) (a>b ? a : b)
void Dataset::read_input(std::string query_filename, std::string target_filename)
{
  query_batch_fasta.open(query_filename, std::ifstream::in);
  if (!query_batch_fasta)
    std::cerr << "something wrong with the files"<< std::endl;

  target_batch_fasta.open(target_filename, std::ifstream::in);
  if (!target_batch_fasta)
    std::cerr << "something wrong with the files"<< std::endl;

  std::string query_batch_line, target_batch_line;

  std::cerr << "Loading files...." << std::endl;

  /*
     Reads FASTA files and fill the corresponding buffers.
     FASTA files contain sequences that are usually on separate lines.
     The file reader detects a '>' then concatenates all the following lines into one sequence, until the next '>' or EOF.
     See more about FASTA format : https://en.wikipedia.org/wiki/FASTA_format
     */

  int seq_begin=0;


  char line_starts[5] = "></+";
  /* The information of reverse-complementing is simulated by changing the first character of the sequence.
   * This is not explicitly FASTA-compliant, although regular FASTA files will simply be interpreted as Forward-Natural direction.
   * From the header of every sequence:
   * - '>' translates to 0b00 (0) = Forward, natural
   * - '<' translates to 0b01 (1) = Reverse, natural
   * - '/' translates to 0b10 (2) = Forward, complemented
   * - '+' translates to 0b11 (3) = Reverse, complemented
   * No protection is done, so any other number will only have its two first bytes counted as above.	 
   */

  while (getline(query_batch_fasta, query_batch_line) && getline(target_batch_fasta, target_batch_line)) { 

    //load sequences from the files
    char *q = NULL;
    char *t = NULL;
    q = strchr(line_starts, (int) (query_batch_line[0]));
    t = strchr(line_starts, (int) (target_batch_line[0]));

    /*  
        t and q are pointers to the first occurence of the first read character in the line_starts array.
        so if I compare the address of these pointers with the address of the pointer to line_start, then...
        I can get which character was found, so which modifier is required. 
JL: for this assignment, we don't consider directions. forward only.
*/

    if (q != NULL && t != NULL) {
      total_seqs++;

      if (seq_begin == 2) {
        // a sequence was already being read. Now it's done, so we should find its length.
        target_seqs_len += (target_seqs.back()).length();
        query_seqs_len += (query_seqs.back()).length();
        maximum_sequence_length = MAX((target_seqs.back()).length(), maximum_sequence_length);
        maximum_sequence_length = MAX((query_seqs.back()).length(), maximum_sequence_length);
      }
      seq_begin = 1;

    } else if (seq_begin == 1) {
      query_seqs.push_back(query_batch_line);
      target_seqs.push_back(target_batch_line);
      seq_begin=2;
    } else if (seq_begin == 2) {
      query_seqs.back() += query_batch_line;
      target_seqs.back() += target_batch_line;
    } else { // should never happen but always put an else, for safety...
      seq_begin = 0;
      std::cerr << "Batch1 and target_batch files should be fasta having same number of sequences" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Check maximum sequence length one more time, to check the last read sequence:
  target_seqs_len += (target_seqs.back()).length();
  query_seqs_len += (query_seqs.back()).length();
  maximum_sequence_length = MAX((target_seqs.back()).length(), maximum_sequence_length);
  maximum_sequence_length = MAX((query_seqs.back()).length(), maximum_sequence_length);
  //int maximum_sequence_length_query = MAX((query_seqs.back()).length(), 0);

#ifdef DEBUG
  std::cerr << "[TEST_PROG DEBUG]: ";
  std::cerr << "Size of read batches are: query=" << query_seqs_len << ", target=" << target_seqs_len << ". maximum_sequence_length=" << maximum_sequence_length << std::endl;
#endif


}
