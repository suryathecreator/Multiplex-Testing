# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 55
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 0.5636 | 0.5818 | 0.6182 | 0.6909 | 0.7818 | 0.8000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 6186.5 | 8409.9 | 12594.5 | 25193.9 | 51798.3 | 104956.9 |

## Detailed Matched-Per-k Summary

- shared_trace_group_4: pass@1 prompts=55, pass@2 prompts=55, pass@4 prompts=55, pass@8 prompts=55, pass@16 prompts=55, pass@32 prompts=55

## Retry Statistics

- shared_trace_group_4: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:03:54 (14634.4 seconds)
- started_at: 2026-05-04T12:40:55.819384+00:00
- finished_at: 2026-05-04T16:44:50.183812+00:00

## Matched Denominators

- pass@1: 55 matched prompts
- pass@2: 55 matched prompts
- pass@4: 55 matched prompts
- pass@8: 55 matched prompts
- pass@16: 55 matched prompts
- pass@32: 55 matched prompts
