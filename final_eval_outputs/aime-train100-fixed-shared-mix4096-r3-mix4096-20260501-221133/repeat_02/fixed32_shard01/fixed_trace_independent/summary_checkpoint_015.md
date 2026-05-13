# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 15
- baseline prompts with full k: 15
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.4000 | 0.4000 | 0.6000 | 0.6000 | 0.7333 | 0.8000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 7348.1 | 14469.1 | 28120.3 | 54799.5 | 110771.5 | 223737.1 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=15, pass@2 prompts=15, pass@4 prompts=15, pass@8 prompts=15, pass@16 prompts=15, pass@32 prompts=15

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 1:27:29 (5248.6 seconds)
- started_at: 2026-05-02T16:59:20.072214+00:00
- finished_at: 2026-05-02T18:26:48.658458+00:00

## Matched Denominators

- pass@1: 15 matched prompts
- pass@2: 15 matched prompts
- pass@4: 15 matched prompts
- pass@8: 15 matched prompts
- pass@16: 15 matched prompts
- pass@32: 15 matched prompts
