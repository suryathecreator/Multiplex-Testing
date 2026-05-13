# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 60
- baseline prompts with full k: 60
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.5000 | 0.5500 | 0.6333 | 0.7000 | 0.7833 | 0.8333 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6485.7 | 13174.3 | 26089.0 | 52564.5 | 104788.4 | 208873.2 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=60, pass@2 prompts=60, pass@4 prompts=60, pass@8 prompts=60, pass@16 prompts=60, pass@32 prompts=60

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:10:58 (18658.1 seconds)
- started_at: 2026-05-02T18:11:58.659958+00:00
- finished_at: 2026-05-02T23:22:56.758088+00:00

## Matched Denominators

- pass@1: 60 matched prompts
- pass@2: 60 matched prompts
- pass@4: 60 matched prompts
- pass@8: 60 matched prompts
- pass@16: 60 matched prompts
- pass@32: 60 matched prompts
