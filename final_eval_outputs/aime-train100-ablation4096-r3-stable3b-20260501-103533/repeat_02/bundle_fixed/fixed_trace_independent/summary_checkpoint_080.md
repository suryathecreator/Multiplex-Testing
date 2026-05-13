# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 80
- baseline prompts with full k: 80
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.4875 | 0.5375 | 0.6125 | 0.6750 | 0.7625 | 0.8125 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6547.0 | 13314.0 | 26450.4 | 53113.5 | 105913.1 | 210648.4 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=80, pass@2 prompts=80, pass@4 prompts=80, pass@8 prompts=80, pass@16 prompts=80, pass@32 prompts=80

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:10:58 (18658.1 seconds)
- started_at: 2026-05-02T18:11:58.659958+00:00
- finished_at: 2026-05-02T23:22:56.758088+00:00

## Matched Denominators

- pass@1: 80 matched prompts
- pass@2: 80 matched prompts
- pass@4: 80 matched prompts
- pass@8: 80 matched prompts
- pass@16: 80 matched prompts
- pass@32: 80 matched prompts
