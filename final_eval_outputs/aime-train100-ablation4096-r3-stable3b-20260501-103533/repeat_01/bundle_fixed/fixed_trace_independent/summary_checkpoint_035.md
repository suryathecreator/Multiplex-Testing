# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 35
- baseline prompts with full k: 35
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.5429 | 0.5714 | 0.7143 | 0.7714 | 0.8286 | 0.8571 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6155.2 | 12786.5 | 25836.2 | 51097.7 | 102747.6 | 203793.2 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=35, pass@2 prompts=35, pass@4 prompts=35, pass@8 prompts=35, pass@16 prompts=35, pass@32 prompts=35

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:22:01 (19320.6 seconds)
- started_at: 2026-05-02T08:24:08.182873+00:00
- finished_at: 2026-05-02T13:46:08.740418+00:00

## Matched Denominators

- pass@1: 35 matched prompts
- pass@2: 35 matched prompts
- pass@4: 35 matched prompts
- pass@8: 35 matched prompts
- pass@16: 35 matched prompts
- pass@32: 35 matched prompts
