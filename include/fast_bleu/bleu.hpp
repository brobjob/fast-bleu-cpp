#ifndef FAST_BLEU_BLEU_HPP_
#define FAST_BLEU_BLEU_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fast_bleu {

using Sentence = std::vector<std::string>;
using Corpus = std::vector<Sentence>;

enum class Smoothing {
  // A missing n-gram precision makes the BLEU score zero.
  kNone,
  // Replace a missing numerator with epsilon, as in NLTK method 1.
  kAddEpsilon,
};

struct Options {
  // One weight per n-gram order. Values must be non-negative and sum to one.
  std::vector<double> weights = {0.25, 0.25, 0.25, 0.25};
  Smoothing smoothing = Smoothing::kAddEpsilon;
  double epsilon = 0.1;

  // Match NLTK's automatic reweighting for the default BLEU-4 weights: for a
  // hypothesis shorter than four tokens, use uniform weights up to its length.
  bool auto_reweight = false;
};

namespace detail {

using Ngram = std::vector<std::string>;

struct NgramHash {
  std::size_t operator()(const Ngram& ngram) const noexcept {
    std::size_t seed = ngram.size();
    for (const auto& token : ngram) {
      const std::size_t value = std::hash<std::string>{}(token);
      seed ^= value + static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) +
              (seed >> 2U);
    }
    return seed;
  }
};

using NgramCounts = std::unordered_map<Ngram, std::size_t, NgramHash>;

struct Precision {
  std::size_t matches = 0;
  std::size_t total = 1;
};

inline NgramCounts count_ngrams(const Sentence& sentence, std::size_t order) {
  NgramCounts counts;
  if (order == 0 || sentence.size() < order) {
    return counts;
  }

  counts.reserve(sentence.size() - order + 1);
  for (std::size_t begin = 0; begin + order <= sentence.size(); ++begin) {
    Ngram ngram(sentence.begin() + static_cast<std::ptrdiff_t>(begin),
                sentence.begin() + static_cast<std::ptrdiff_t>(begin + order));
    ++counts[std::move(ngram)];
  }
  return counts;
}

inline Options validate_options(Options options) {
  if (options.weights.empty()) {
    throw std::invalid_argument("BLEU weights must not be empty");
  }

  double sum = 0.0;
  for (double weight : options.weights) {
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument(
          "BLEU weights must be finite and non-negative");
    }
    sum += weight;
  }
  if (!std::isfinite(sum) || std::abs(sum - 1.0) > 1e-12) {
    throw std::invalid_argument("BLEU weights must sum to one");
  }
  if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0) {
    throw std::invalid_argument("BLEU epsilon must be finite and positive");
  }
  return options;
}

inline bool has_default_bleu4_weights(const std::vector<double>& weights) {
  if (weights.size() != 4) {
    return false;
  }
  return std::all_of(weights.begin(), weights.end(),
                     [](double weight) { return weight == 0.25; });
}

inline std::vector<double> effective_weights(const Options& options,
                                             std::size_t hypothesis_length) {
  if (!options.auto_reweight || hypothesis_length == 0 ||
      hypothesis_length >= 4 || !has_default_bleu4_weights(options.weights)) {
    return options.weights;
  }
  return std::vector<double>(hypothesis_length,
                             1.0 / static_cast<double>(hypothesis_length));
}

inline std::size_t distance(std::size_t left, std::size_t right) {
  return left > right ? left - right : right - left;
}

inline std::size_t closest_length(const std::vector<std::size_t>& lengths,
                                  std::size_t hypothesis_length) {
  if (lengths.empty()) {
    throw std::invalid_argument("BLEU requires at least one reference");
  }

  std::size_t closest = lengths.front();
  for (std::size_t length : lengths) {
    const auto candidate =
        std::make_pair(distance(length, hypothesis_length), length);
    const auto current =
        std::make_pair(distance(closest, hypothesis_length), closest);
    if (candidate < current) {
      closest = length;
    }
  }
  return closest;
}

inline double brevity_penalty(std::size_t reference_length,
                              std::size_t hypothesis_length) {
  if (hypothesis_length == 0) {
    return 0.0;
  }
  if (hypothesis_length > reference_length) {
    return 1.0;
  }
  return std::exp(1.0 - static_cast<double>(reference_length) /
                            static_cast<double>(hypothesis_length));
}

inline double combine(const std::vector<Precision>& precisions,
                      const std::vector<double>& weights,
                      const Options& options, double brevity) {
  // NLTK and fast-bleu both define zero unigram overlap as a hard zero,
  // regardless of the selected smoothing method.
  if (precisions.front().matches == 0) {
    return 0.0;
  }

  long double weighted_log_sum = 0.0L;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    if (weights[index] == 0.0) {
      continue;
    }

    const Precision precision = precisions[index];
    long double value = 0.0L;
    if (precision.matches == 0) {
      if (options.smoothing == Smoothing::kNone) {
        return 0.0;
      }
      value = static_cast<long double>(options.epsilon) /
              static_cast<long double>(precision.total);
    } else {
      value = static_cast<long double>(precision.matches) /
              static_cast<long double>(precision.total);
    }
    weighted_log_sum +=
        static_cast<long double>(weights[index]) * std::log(value);
  }
  return brevity * static_cast<double>(std::exp(weighted_log_sum));
}

class ReferenceStatistics {
 public:
  ReferenceStatistics(const Corpus& references, std::size_t max_order)
      : max_counts_(max_order) {
    if (references.empty()) {
      throw std::invalid_argument("BLEU requires at least one reference");
    }

    lengths_.reserve(references.size());
    for (const auto& reference : references) {
      lengths_.push_back(reference.size());
      for (std::size_t order = 1; order <= max_order; ++order) {
        for (const auto& item : count_ngrams(reference, order)) {
          auto& maximum = max_counts_[order - 1][item.first];
          maximum = std::max(maximum, item.second);
        }
      }
    }
  }

  Precision precision(const Sentence& hypothesis, std::size_t order) const {
    const NgramCounts counts = count_ngrams(hypothesis, order);
    Precision result;
    result.total = std::max<std::size_t>(
        1, hypothesis.size() >= order ? hypothesis.size() - order + 1 : 0);

    const auto& reference_counts = max_counts_.at(order - 1);
    for (const auto& item : counts) {
      const auto reference = reference_counts.find(item.first);
      if (reference != reference_counts.end()) {
        result.matches += std::min(item.second, reference->second);
      }
    }
    return result;
  }

  std::size_t closest_length(std::size_t hypothesis_length) const {
    return detail::closest_length(lengths_, hypothesis_length);
  }

 private:
  std::vector<NgramCounts> max_counts_;
  std::vector<std::size_t> lengths_;
};

struct TopTwoCounts {
  std::size_t largest = 0;
  std::size_t second_largest = 0;
  std::size_t largest_frequency = 0;

  void add(std::size_t count) {
    if (count > largest) {
      second_largest = largest;
      largest = count;
      largest_frequency = 1;
    } else if (count == largest) {
      ++largest_frequency;
    } else if (count > second_largest) {
      second_largest = count;
    }
  }
};

using TopNgramCounts = std::unordered_map<Ngram, TopTwoCounts, NgramHash>;

}  // namespace detail

// Scores hypotheses against a fixed reference set. Reference n-gram maxima are
// computed once and reused, making score_all suitable for large batches.
class BleuScorer {
 public:
  explicit BleuScorer(const Corpus& references, Options options = {})
      : options_(detail::validate_options(std::move(options))),
        references_(references, options_.weights.size()) {}

  double score(const Sentence& hypothesis) const {
    if (hypothesis.empty()) {
      return 0.0;
    }

    const std::vector<double> weights =
        detail::effective_weights(options_, hypothesis.size());
    std::vector<detail::Precision> precisions;
    precisions.reserve(weights.size());
    for (std::size_t order = 1; order <= weights.size(); ++order) {
      precisions.push_back(references_.precision(hypothesis, order));
    }

    const double brevity = detail::brevity_penalty(
        references_.closest_length(hypothesis.size()), hypothesis.size());
    return detail::combine(precisions, weights, options_, brevity);
  }

  std::vector<double> score_all(const Corpus& hypotheses) const {
    std::vector<double> results;
    results.reserve(hypotheses.size());
    for (const auto& hypothesis : hypotheses) {
      results.push_back(score(hypothesis));
    }
    return results;
  }

 private:
  Options options_;
  detail::ReferenceStatistics references_;
};

inline double sentence_bleu(const Corpus& references,
                            const Sentence& hypothesis, Options options = {}) {
  return BleuScorer(references, std::move(options)).score(hypothesis);
}

// Computes leave-one-out SelfBLEU. Per-sentence n-gram counts and the two
// largest counts across the corpus are cached so excluding one sentence does
// not require rebuilding a scorer for every item.
class SelfBleuScorer {
 public:
  explicit SelfBleuScorer(Corpus sentences, Options options = {})
      : options_(detail::validate_options(std::move(options))),
        sentences_(std::move(sentences)),
        sentence_counts_(sentences_.size()),
        top_counts_(options_.weights.size()) {
    if (sentences_.size() < 2) {
      throw std::invalid_argument("SelfBLEU requires at least two sentences");
    }

    for (std::size_t sentence_index = 0; sentence_index < sentences_.size();
         ++sentence_index) {
      auto& counts_by_order = sentence_counts_[sentence_index];
      counts_by_order.reserve(options_.weights.size());
      for (std::size_t order = 1; order <= options_.weights.size(); ++order) {
        counts_by_order.push_back(
            detail::count_ngrams(sentences_[sentence_index], order));
        for (const auto& item : counts_by_order.back()) {
          top_counts_[order - 1][item.first].add(item.second);
        }
      }
    }
  }

  double score(std::size_t sentence_index) const {
    if (sentence_index >= sentences_.size()) {
      throw std::out_of_range("SelfBLEU sentence index is out of range");
    }
    const Sentence& hypothesis = sentences_[sentence_index];
    if (hypothesis.empty()) {
      return 0.0;
    }

    const std::vector<double> weights =
        detail::effective_weights(options_, hypothesis.size());
    std::vector<detail::Precision> precisions;
    precisions.reserve(weights.size());
    for (std::size_t order = 1; order <= weights.size(); ++order) {
      detail::Precision precision;
      precision.total = std::max<std::size_t>(
          1, hypothesis.size() >= order ? hypothesis.size() - order + 1 : 0);

      const auto& hypothesis_counts =
          sentence_counts_[sentence_index][order - 1];
      const auto& aggregate = top_counts_[order - 1];
      for (const auto& item : hypothesis_counts) {
        const detail::TopTwoCounts& top = aggregate.at(item.first);
        std::size_t reference_max = top.largest;
        if (item.second == top.largest && top.largest_frequency == 1) {
          reference_max = top.second_largest;
        }
        precision.matches += std::min(item.second, reference_max);
      }
      precisions.push_back(precision);
    }

    std::vector<std::size_t> reference_lengths;
    reference_lengths.reserve(sentences_.size() - 1);
    for (std::size_t index = 0; index < sentences_.size(); ++index) {
      if (index != sentence_index) {
        reference_lengths.push_back(sentences_[index].size());
      }
    }
    const double brevity = detail::brevity_penalty(
        detail::closest_length(reference_lengths, hypothesis.size()),
        hypothesis.size());
    return detail::combine(precisions, weights, options_, brevity);
  }

  std::vector<double> scores() const {
    std::vector<double> results;
    results.reserve(sentences_.size());
    for (std::size_t index = 0; index < sentences_.size(); ++index) {
      results.push_back(score(index));
    }
    return results;
  }

 private:
  Options options_;
  Corpus sentences_;
  std::vector<std::vector<detail::NgramCounts>> sentence_counts_;
  std::vector<detail::TopNgramCounts> top_counts_;
};

inline std::vector<double> self_bleu(const Corpus& sentences,
                                     Options options = {}) {
  return SelfBleuScorer(sentences, std::move(options)).scores();
}

}  // namespace fast_bleu

#endif  // FAST_BLEU_BLEU_HPP_
