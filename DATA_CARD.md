# Data Card: Hindi–Hinglish Homophobic Hate-Speech Counter-Narratives

## Summary

| | |
| --- | --- |
| Size | 1,550 pairs |
| Languages | Hindi (Devanagari) and Hinglish (Hindi–English code-mixed, mostly Roman script) |
| Fields | `ID`, `Text` (hate speech or control comment), `Category`, `Target`, `Counter Narrative` |
| Categories | Homophobic 1,404 (90.6%), Non-Homophobic 146 (9.4%, control class) |
| Targets | LGBTQ+ Community 214, Gay Men 214, Transgender Individuals 209, Lesbians 206, Bisexual Individuals 202, Same-sex Marriage 184, Pride Community 178, No Target 143 |
| Script of `Text` | Romanized Hinglish 991 (63.9%), Devanagari 559 (36.1%) |
| Mean length | 15 words (hate speech), 24 words (counter-narrative) |
| Licence | CC BY-NC-SA 4.0 (research and non-commercial use) |

## How the data was created

1. **Seed (V1, 50 pairs).** Comments from the LT-EDI Hindi homophobia and
   transphobia data, originally collected from social media. Names, user
   handles and links were removed, and a counter-narrative was written by hand
   for each comment.
2. **Generator choice.** GPT, Claude, Grok and Gemini each produced candidate
   V2 pairs from the same seed and instructions. GPT scored highest on BLEU-1,
   BLEU-2, METEOR and ROUGE-L and was judged the most natural in Hinglish, so
   it was used for all later rounds.
3. **Versioned expansion.** Rounds V2–V5 added 100, 200, 400 and 800 pairs
   (cumulative 150, 350, 750, 1,550). Each round was conditioned on the
   reviewed data of the previous round.
4. **Human review.** Every generated pair was reviewed. Pairs were accepted
   unchanged or post-edited; no pair was discarded. The generated and edited
   versions are both kept in `data/versions/*_m.xlsx`.

## Quality per round

| Round | New pairs | Accepted unchanged | HTER (all pairs) | Novelty vs V1 (H / CN) |
| --- | --- | --- | --- | --- |
| V2 | 100 | 89.0% | 0.008 | 0.904 / 0.909 |
| V3 | 200 | 70.0% | 0.045 | 0.930 / 0.916 |
| V4 | 400 | 78.0% | 0.080 | 0.961 / 0.960 |
| V5 | 800 | 90.5% | 0.032 | 0.931 / 0.911 |

Full per-round statistics are in `data/versions/metrics.xlsx` and in the paper.

## Known properties (please read before use)

- **Reply frames.** Content-word trigram repetition is higher in V4–V5 than in
  V2–V3, so later rounds reuse a smaller set of reply frames.
- **Control class.** Most non-homophobic pairs come from rounds V2–V4.
- **Duplicates.** 22 counter-narratives occur more than once; all hate-speech
  texts are unique.
- **Target labels.** Rows labelled `None` in the working files were merged
  into `No Target` in this release.
- **Evaluation set.** `data/eval_test141.jsonl` holds the 141 homophobic posts
  used in the paper. It covers six of the seven target groups.
- **Model influence.** Expansion used GPT, so phrasing may reflect that model;
  human review corrected unnatural or unsafe replies.

## Intended use

Research on counter-narrative generation, evaluation of grounded generation,
retrieval for code-mixed text, and moderator-assist tools. Not for training
systems that post replies automatically without human review, and not for
generating hate speech.

## Ethics

The data contains offensive language about LGBTQ+ people, included only for
research. No real person is referenced in the expanded text. Reviewers were
told they would read offensive content and could stop at any time.
