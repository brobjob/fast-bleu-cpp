#include <iostream>

#include "fast_bleu/bleu.hpp"

int main() {
  const fast_bleu::Corpus references = {
      {"the", "cat", "is", "on", "the", "mat"},
      {"there", "is", "a", "cat", "on", "the", "mat"},
  };
  const fast_bleu::Sentence hypothesis = {"the", "cat", "is",
                                          "on",  "the", "mat"};

  const fast_bleu::BleuScorer scorer(references);
  std::cout << scorer.score(hypothesis) << '\n';
  return 0;
}
