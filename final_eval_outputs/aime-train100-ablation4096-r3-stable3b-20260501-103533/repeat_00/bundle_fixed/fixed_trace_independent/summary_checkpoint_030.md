# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 30
- baseline prompts with full k: 30
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.6333 | 0.7333 | 0.8000 | 0.8333 | 0.8667 | 0.8667 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 5999.1 | 11920.5 | 24031.2 | 47902.7 | 97499.0 | 195773.4 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=30, pass@2 prompts=30, pass@4 prompts=30, pass@8 prompts=30, pass@16 prompts=30, pass@32 prompts=30

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:40:51 (20450.5 seconds)
- started_at: 2026-05-02T04:51:02.787718+00:00
- finished_at: 2026-05-02T10:31:53.295001+00:00

## Matched Denominators

- pass@1: 30 matched prompts
- pass@2: 30 matched prompts
- pass@4: 30 matched prompts
- pass@8: 30 matched prompts
- pass@16: 30 matched prompts
- pass@32: 30 matched prompts
