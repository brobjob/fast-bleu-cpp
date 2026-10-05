# fast-bleu-cpp

`fast-bleu-cpp` is a dependency-free, header-only C++17 library for computing
BLEU and SelfBLEU scores over pre-tokenized text. It is inspired by
[fast-bleu](https://github.com/Danial-Alh/fast-bleu), while providing a native
C++ API without Python, OpenMP, or platform-specific runtime dependencies.

The library caches reference n-gram statistics for repeated scoring. Its
SelfBLEU implementation also caches per-sentence counts and the two largest
reference counts, so each leave-one-out score does not rebuild the reference
set.

## Use with Bazel

Add the module to `MODULE.bazel`:

```starlark
bazel_dep(name = "fast-bleu-cpp", version = "0.1.0")
```

Then depend on its public target:

```starlark
cc_binary(
    name = "app",
    srcs = ["app.cc"],
    deps = ["@fast-bleu-cpp"],
)
```

Your C++ target must compile as C++17 or newer.

## Score a hypothesis

```cpp
#include "fast_bleu/bleu.hpp"

#include <iostream>

int main() {
  const fast_bleu::Corpus references = {
      {"the", "cat", "is", "on", "the", "mat"},
      {"there", "is", "a", "cat", "on", "the", "mat"},
  };
  const fast_bleu::Sentence hypothesis =
      {"the", "cat", "is", "on", "the", "mat"};

  const fast_bleu::BleuScorer scorer(references);
  std::cout << scorer.score(hypothesis) << '\n';
}
```

`BleuScorer` is immutable after construction and can be reused to score a batch:

```cpp
const fast_bleu::Corpus hypotheses = {hypothesis_a, hypothesis_b};
const std::vector<double> scores = scorer.score_all(hypotheses);
```

For a one-off calculation, use `fast_bleu::sentence_bleu`.

## Configure BLEU

The defaults are uniform BLEU-4 weights and NLTK method-1-style smoothing:

```cpp
fast_bleu::Options options;
options.weights = {0.5, 0.5};  // BLEU-2
options.smoothing = fast_bleu::Smoothing::kAddEpsilon;
options.epsilon = 0.1;

const fast_bleu::BleuScorer scorer(references, options);
```

Set smoothing to `kNone` when any missing weighted n-gram should make the score
zero. Set `auto_reweight` to reproduce NLTK's short-hypothesis behavior for the
default BLEU-4 weights.

Scores use standard modified n-gram precision, closest-reference length with a
shorter-reference tie break, and the standard brevity penalty. Token boundaries
are represented structurally, so tokens containing spaces cannot create an
n-gram collision.

## Compute SelfBLEU

```cpp
const fast_bleu::SelfBleuScorer scorer(sentences, options);
const std::vector<double> scores = scorer.scores();
```

Each sentence is scored against every other sentence. At least two sentences
are required.

## Scope

- Inputs must already be tokenized; the library does not choose a tokenizer.
- `BleuScorer` computes sentence BLEU against one fixed reference set. It does
  not aggregate counts across aligned segments as corpus BLEU does.
- Smoothing modes are no smoothing and NLTK method 1.
- All public calculations use `double` precision. This produces tiny differences
  from fast-bleu's published trigram examples, whose C++ boundary narrows weights
  to `float`; results match NLTK's double-precision calculation.

## Build and test

```console
bazel test //...
bazel run //:example
```

The CI matrix covers Bazel 8 and 9 on Linux, macOS, and Windows.

## Attribution

The API and implementation are new, but the fixed-reference and SelfBLEU use
cases are based on Danial Alihosseini's
[fast-bleu](https://github.com/Danial-Alh/fast-bleu) project and its associated
[paper](https://aclanthology.org/W19-2311/), *Jointly Measuring Diversity and
Quality in Text Generation Models*. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT
