# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 70
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 0.5286 | 0.5571 | 0.6000 | 0.6571 | 0.7429 | 0.7571 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 6275.4 | 8575.3 | 13006.8 | 26379.0 | 54107.4 | 108975.8 |

## Detailed Matched-Per-k Summary

- shared_trace_group_4: pass@1 prompts=70, pass@2 prompts=70, pass@4 prompts=70, pass@8 prompts=70, pass@16 prompts=70, pass@32 prompts=70

## Retry Statistics

- shared_trace_group_4: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:03:54 (14634.4 seconds)
- started_at: 2026-05-04T12:40:55.819384+00:00
- finished_at: 2026-05-04T16:44:50.183812+00:00

## Matched Denominators

- pass@1: 70 matched prompts
- pass@2: 70 matched prompts
- pass@4: 70 matched prompts
- pass@8: 70 matched prompts
- pass@16: 70 matched prompts
- pass@32: 70 matched prompts
