#include <cmath>
#include <cstdlib>

#include "fast_bleu/bleu.hpp"

int main() {
  const fast_bleu::Corpus references = {
      {"the", "cat", "is", "on", "the", "mat"},
      {"there", "is", "a", "cat", "on", "the", "mat"},
  };
  const fast_bleu::Sentence hypothesis = {"the", "cat", "is",
                                          "on",  "the", "mat"};

  const double score = fast_bleu::sentence_bleu(references, hypothesis);
  return std::abs(score - 1.0) < 1e-12 ? EXIT_SUCCESS : EXIT_FAILURE;
}
