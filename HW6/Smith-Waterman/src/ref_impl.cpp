#include "ref_impl.h"





void mgp_opt_align(Parameters *args, Dataset *dataset)
{
    Algorithms::SimilarityAlgorithmParallel *parallel = new Algorithms::SmithWatermanParallel(dataset->maximum_sequence_length, args->sa, args->sb, args->gapo, args->gape);

  for(unsigned int i=0;i<dataset->target_seqs.size();i++)
  {
   // printf("processing sequence pair %d\n", i);
    printf("Seq %d ", i);
    parallel->setSeq1(dataset->target_seqs[i].c_str(), dataset->target_seqs[i].size());
    parallel->setSeq2(dataset->query_seqs[i].c_str(), dataset->query_seqs[i].size());
    parallel->Run();
  }

}

