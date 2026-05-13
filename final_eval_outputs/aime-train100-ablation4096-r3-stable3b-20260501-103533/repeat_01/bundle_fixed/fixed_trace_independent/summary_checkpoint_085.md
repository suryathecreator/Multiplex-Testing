# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 85
- baseline prompts with full k: 85
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.4824 | 0.5294 | 0.6353 | 0.6941 | 0.7294 | 0.7529 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6500.8 | 13112.0 | 26451.7 | 52598.7 | 105606.8 | 210023.8 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=85, pass@2 prompts=85, pass@4 prompts=85, pass@8 prompts=85, pass@16 prompts=85, pass@32 prompts=85

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:22:01 (19320.6 seconds)
- started_at: 2026-05-02T08:24:08.182873+00:00
- finished_at: 2026-05-02T13:46:08.740418+00:00

## Matched Denominators

- pass@1: 85 matched prompts
- pass@2: 85 matched prompts
- pass@4: 85 matched prompts
- pass@8: 85 matched prompts
- pass@16: 85 matched prompts
- pass@32: 85 matched prompts
