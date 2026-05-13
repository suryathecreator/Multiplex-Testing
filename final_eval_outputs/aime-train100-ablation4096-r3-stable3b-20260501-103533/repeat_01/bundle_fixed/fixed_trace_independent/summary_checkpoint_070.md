# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 70
- baseline prompts with full k: 70
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.5143 | 0.5571 | 0.6714 | 0.7286 | 0.7714 | 0.8000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6376.3 | 12945.8 | 26224.2 | 52347.5 | 105097.1 | 209246.4 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=70, pass@2 prompts=70, pass@4 prompts=70, pass@8 prompts=70, pass@16 prompts=70, pass@32 prompts=70

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:22:01 (19320.6 seconds)
- started_at: 2026-05-02T08:24:08.182873+00:00
- finished_at: 2026-05-02T13:46:08.740418+00:00

## Matched Denominators

- pass@1: 70 matched prompts
- pass@2: 70 matched prompts
- pass@4: 70 matched prompts
- pass@8: 70 matched prompts
- pass@16: 70 matched prompts
- pass@32: 70 matched prompts
