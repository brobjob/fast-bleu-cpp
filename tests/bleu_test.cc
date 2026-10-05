#include "fast_bleu/bleu.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* expression, int line) {
  if (!condition) {
    std::cerr << "line " << line << ": expectation failed: " << expression
              << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, double tolerance,
                 const char* expression, int line) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::cerr << std::setprecision(17) << "line " << line << ": " << expression
              << " was " << actual << ", expected " << expected << " +/- "
              << tolerance << '\n';
    ++failures;
  }
}

#define EXPECT(...) expect((__VA_ARGS__), #__VA_ARGS__, __LINE__)
#define EXPECT_NEAR(actual, expected, tolerance) \
  expect_near((actual), (expected), (tolerance), #actual, __LINE__)

fast_bleu::Options options(
    std::vector<double> weights,
    fast_bleu::Smoothing smoothing = fast_bleu::Smoothing::kAddEpsilon) {
  fast_bleu::Options result;
  result.weights = std::move(weights);
  result.smoothing = smoothing;
  return result;
}

const fast_bleu::Corpus& example_references() {
  static const fast_bleu::Corpus references = {
      {"It", "is", "a", "guide", "to", "action", "that", "ensures", "that",
       "the", "military", "will", "forever", "heed", "Party", "commands"},
      {"It", "is", "the", "guiding", "principle", "which", "guarantees", "the",
       "military", "forces", "always", "being", "under", "the", "command", "of",
       "the", "Party"},
      {"It", "is", "the", "practical", "guide", "for", "the", "army", "always",
       "to", "heed", "the", "directions", "of", "the", "party"},
  };
  return references;
}

const fast_bleu::Corpus& example_hypotheses() {
  static const fast_bleu::Corpus hypotheses = {
      {"It", "is", "a", "guide", "to", "action", "which", "ensures", "that",
       "the", "military", "always", "obeys", "the", "commands", "of", "the",
       "party"},
      {"he", "read", "the", "book", "because", "he", "was", "interested", "in",
       "world", "history"},
  };
  return hypotheses;
}

void matches_reference_bleu_examples() {
  const fast_bleu::BleuScorer bigram(example_references(), options({0.5, 0.5}));
  const auto bigram_scores = bigram.score_all(example_hypotheses());
  EXPECT(bigram_scores.size() == 2);
  EXPECT_NEAR(bigram_scores[0], 0.7453559924999299, 1e-12);
  EXPECT_NEAR(bigram_scores[1], 0.0191380231127159, 1e-12);

  const fast_bleu::BleuScorer trigram(example_references(),
                                      options({1.0 / 3, 1.0 / 3, 1.0 / 3}));
  const auto trigram_scores = trigram.score_all(example_hypotheses());
  // fast-bleu publishes nearby values because its native boundary narrows
  // weights to float. These values match NLTK's double-precision calculation.
  EXPECT_NEAR(trigram_scores[0], 0.6240726989348756, 1e-12);
  EXPECT_NEAR(trigram_scores[1], 0.01372087114384104, 1e-12);
}

void computes_identity_clipping_and_brevity() {
  EXPECT_NEAR(fast_bleu::sentence_bleu({{"a", "small", "exact", "example"}},
                                       {"a", "small", "exact", "example"}),
              1.0, 1e-12);

  EXPECT_NEAR(
      fast_bleu::sentence_bleu({{"the", "cat"}}, {"the", "the", "the"},
                               options({1.0}, fast_bleu::Smoothing::kNone)),
      1.0 / 3.0, 1e-12);

  EXPECT_NEAR(
      fast_bleu::sentence_bleu({{"a", "b", "c", "d"}}, {"a", "b"},
                               options({1.0}, fast_bleu::Smoothing::kNone)),
      std::exp(-1.0), 1e-12);
}

void applies_smoothing_and_auto_reweighting() {
  EXPECT_NEAR(
      fast_bleu::sentence_bleu({{"a", "b"}}, {"a", "c"}, options({0.5, 0.5})),
      std::sqrt(0.05), 1e-12);

  EXPECT_NEAR(fast_bleu::sentence_bleu(
                  {{"a", "b"}}, {"a", "c"},
                  options({0.5, 0.5}, fast_bleu::Smoothing::kNone)),
              0.0, 0.0);

  // Method-1 smoothing does not manufacture a score without unigram overlap.
  EXPECT_NEAR(
      fast_bleu::sentence_bleu({{"a", "b"}}, {"x", "y"}, options({0.5, 0.5})),
      0.0, 0.0);

  fast_bleu::Options automatic;
  automatic.auto_reweight = true;
  automatic.smoothing = fast_bleu::Smoothing::kNone;
  EXPECT_NEAR(fast_bleu::sentence_bleu({{"a", "b"}}, {"a", "b"}, automatic),
              1.0, 1e-12);

  fast_bleu::Options almost_default;
  almost_default.weights = {0.2500000000005, 0.2499999999995, 0.25, 0.25};
  almost_default.auto_reweight = true;
  EXPECT_NEAR(
      fast_bleu::sentence_bleu({{"a", "b"}}, {"a", "b"}, almost_default),
      std::sqrt(0.1), 1e-12);
}

void keeps_token_boundaries_unambiguous() {
  const double score = fast_bleu::sentence_bleu(
      {{"shared", "a", "b c"}}, {"shared", "a b", "c"},
      options({0.0, 1.0}, fast_bleu::Smoothing::kNone));
  EXPECT_NEAR(score, 0.0, 0.0);
}

void computes_self_bleu_without_rebuilding_references() {
  const auto bigram =
      fast_bleu::self_bleu(example_references(), options({0.5, 0.5}));
  EXPECT(bigram.size() == 3);
  EXPECT_NEAR(bigram[0], 0.25819888974716115, 1e-12);
  EXPECT_NEAR(bigram[1], 0.3615507630310936, 1e-12);
  EXPECT_NEAR(bigram[2], 0.37080992435478316, 1e-12);

  const auto trigram = fast_bleu::self_bleu(
      example_references(), options({1.0 / 3, 1.0 / 3, 1.0 / 3}));
  EXPECT_NEAR(trigram[0], 0.07808966656190802, 1e-12);
  EXPECT_NEAR(trigram[1], 0.20140621167558334, 1e-12);
  EXPECT_NEAR(trigram[2], 0.2141533574180157, 1e-12);

  const fast_bleu::Corpus repeated = {
      {"a", "a", "a", "b"},
      {"a", "a", "b", "b"},
      {"a", "b", "b", "b"},
      {"x", "y", "z"},
  };
  const auto config = options({0.5, 0.5});
  const fast_bleu::SelfBleuScorer optimized(repeated, config);
  for (std::size_t held_out = 0; held_out < repeated.size(); ++held_out) {
    fast_bleu::Corpus references;
    for (std::size_t index = 0; index < repeated.size(); ++index) {
      if (index != held_out) {
        references.push_back(repeated[index]);
      }
    }
    const fast_bleu::BleuScorer direct(references, config);
    EXPECT_NEAR(optimized.score(held_out), direct.score(repeated[held_out]),
                1e-12);
  }
}

void handles_empty_input_and_rejects_invalid_configuration() {
  EXPECT_NEAR(fast_bleu::sentence_bleu({{"reference"}}, {}), 0.0, 0.0);

  bool threw = false;
  try {
    static_cast<void>(fast_bleu::BleuScorer({}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);

  threw = false;
  try {
    fast_bleu::Options invalid;
    invalid.weights = {0.2, 0.2};
    static_cast<void>(fast_bleu::BleuScorer({{"reference"}}, invalid));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);

  threw = false;
  try {
    static_cast<void>(fast_bleu::SelfBleuScorer({{"only one"}}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

}  // namespace

int main() {
  matches_reference_bleu_examples();
  computes_identity_clipping_and_brevity();
  applies_smoothing_and_auto_reweighting();
  keeps_token_boundaries_unambiguous();
  computes_self_bleu_without_rebuilding_references();
  handles_empty_input_and_rejects_invalid_configuration();

  if (failures != 0) {
    std::cerr << failures << " test expectation(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "all fast-bleu-cpp tests passed\n";
  return EXIT_SUCCESS;
}
