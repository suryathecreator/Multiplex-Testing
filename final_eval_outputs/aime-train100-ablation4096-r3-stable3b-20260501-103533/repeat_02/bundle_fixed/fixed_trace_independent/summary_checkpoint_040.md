# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 40
- baseline prompts with full k: 40
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.4750 | 0.5500 | 0.6250 | 0.7250 | 0.8000 | 0.8250 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6723.1 | 13468.6 | 26317.5 | 52731.8 | 105381.0 | 210472.5 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=40, pass@2 prompts=40, pass@4 prompts=40, pass@8 prompts=40, pass@16 prompts=40, pass@32 prompts=40

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:10:58 (18658.1 seconds)
- started_at: 2026-05-02T18:11:58.659958+00:00
- finished_at: 2026-05-02T23:22:56.758088+00:00

## Matched Denominators

- pass@1: 40 matched prompts
- pass@2: 40 matched prompts
- pass@4: 40 matched prompts
- pass@8: 40 matched prompts
- pass@16: 40 matched prompts
- pass@32: 40 matched prompts
