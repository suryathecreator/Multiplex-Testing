# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 20
- baseline prompts with full k: 20
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.3500 | 0.4500 | 0.4500 | 0.5000 | 0.7000 | 0.8500 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6781.1 | 13630.5 | 26917.0 | 54649.4 | 107495.1 | 213318.5 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=20, pass@2 prompts=20, pass@4 prompts=20, pass@8 prompts=20, pass@16 prompts=20, pass@32 prompts=20

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 1:27:15 (5235.3 seconds)
- started_at: 2026-05-02T13:45:57.215345+00:00
- finished_at: 2026-05-02T15:13:12.522372+00:00

## Matched Denominators

- pass@1: 20 matched prompts
- pass@2: 20 matched prompts
- pass@4: 20 matched prompts
- pass@8: 20 matched prompts
- pass@16: 20 matched prompts
- pass@32: 20 matched prompts
