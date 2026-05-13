# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 15
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 0.8667 | 0.8667 | 0.8667 | 0.8667 | 0.8667 | 0.8667 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 5528.2 | 6952.9 | 9322.8 | 19104.5 | 39756.1 | 79552.7 |

## Detailed Matched-Per-k Summary

- shared_trace_group_4: pass@1 prompts=15, pass@2 prompts=15, pass@4 prompts=15, pass@8 prompts=15, pass@16 prompts=15, pass@32 prompts=15

## Retry Statistics

- shared_trace_group_4: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:03:54 (14634.4 seconds)
- started_at: 2026-05-04T12:40:55.819384+00:00
- finished_at: 2026-05-04T16:44:50.183812+00:00

## Matched Denominators

- pass@1: 15 matched prompts
- pass@2: 15 matched prompts
- pass@4: 15 matched prompts
- pass@8: 15 matched prompts
- pass@16: 15 matched prompts
- pass@32: 15 matched prompts
