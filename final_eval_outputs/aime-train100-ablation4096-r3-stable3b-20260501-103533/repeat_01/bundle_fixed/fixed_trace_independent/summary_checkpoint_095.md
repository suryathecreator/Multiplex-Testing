# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 95
- baseline prompts with full k: 95
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.4947 | 0.5474 | 0.6421 | 0.6947 | 0.7263 | 0.7474 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6516.1 | 13116.5 | 26357.8 | 52544.0 | 105431.5 | 209894.6 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=95, pass@2 prompts=95, pass@4 prompts=95, pass@8 prompts=95, pass@16 prompts=95, pass@32 prompts=95

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:22:01 (19320.6 seconds)
- started_at: 2026-05-02T08:24:08.182873+00:00
- finished_at: 2026-05-02T13:46:08.740418+00:00

## Matched Denominators

- pass@1: 95 matched prompts
- pass@2: 95 matched prompts
- pass@4: 95 matched prompts
- pass@8: 95 matched prompts
- pass@16: 95 matched prompts
- pass@32: 95 matched prompts
