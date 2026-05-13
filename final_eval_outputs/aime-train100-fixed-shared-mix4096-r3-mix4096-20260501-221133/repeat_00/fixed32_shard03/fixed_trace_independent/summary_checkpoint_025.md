# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 25
- baseline prompts with full k: 25
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.3200 | 0.4000 | 0.4400 | 0.4400 | 0.5200 | 0.6400 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 7231.3 | 14070.6 | 27467.8 | 54383.8 | 108738.6 | 219009.0 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=25, pass@2 prompts=25, pass@4 prompts=25, pass@8 prompts=25, pass@16 prompts=25, pass@32 prompts=25

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 1:30:54 (5454.2 seconds)
- started_at: 2026-05-02T10:52:31.003196+00:00
- finished_at: 2026-05-02T12:23:25.155097+00:00

## Matched Denominators

- pass@1: 25 matched prompts
- pass@2: 25 matched prompts
- pass@4: 25 matched prompts
- pass@8: 25 matched prompts
- pass@16: 25 matched prompts
- pass@32: 25 matched prompts
