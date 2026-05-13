# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 30
- baseline prompts with full k: 30
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.5333 | 0.6000 | 0.6667 | 0.7667 | 0.8000 | 0.8333 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6503.6 | 12886.0 | 25231.5 | 50425.7 | 101336.8 | 202435.4 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=30, pass@2 prompts=30, pass@4 prompts=30, pass@8 prompts=30, pass@16 prompts=30, pass@32 prompts=30

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:10:58 (18658.1 seconds)
- started_at: 2026-05-02T18:11:58.659958+00:00
- finished_at: 2026-05-02T23:22:56.758088+00:00

## Matched Denominators

- pass@1: 30 matched prompts
- pass@2: 30 matched prompts
- pass@4: 30 matched prompts
- pass@8: 30 matched prompts
- pass@16: 30 matched prompts
- pass@32: 30 matched prompts
