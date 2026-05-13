# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 40
- baseline prompts with full k: 40
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.5750 | 0.6750 | 0.7250 | 0.8000 | 0.8500 | 0.8500 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6302.2 | 12542.4 | 25335.0 | 50350.9 | 102621.9 | 205685.0 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=40, pass@2 prompts=40, pass@4 prompts=40, pass@8 prompts=40, pass@16 prompts=40, pass@32 prompts=40

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:40:51 (20450.5 seconds)
- started_at: 2026-05-02T04:51:02.787718+00:00
- finished_at: 2026-05-02T10:31:53.295001+00:00

## Matched Denominators

- pass@1: 40 matched prompts
- pass@2: 40 matched prompts
- pass@4: 40 matched prompts
- pass@8: 40 matched prompts
- pass@16: 40 matched prompts
- pass@32: 40 matched prompts
