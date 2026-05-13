# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 15
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 0.8000 | 0.8000 | 0.8000 | 0.8667 | 0.8667 | 0.8667 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 5416.2 | 6867.8 | 9623.9 | 19954.2 | 39856.3 | 81744.3 |

## Detailed Matched-Per-k Summary

- shared_trace_group_4: pass@1 prompts=15, pass@2 prompts=15, pass@4 prompts=15, pass@8 prompts=15, pass@16 prompts=15, pass@32 prompts=15

## Retry Statistics

- shared_trace_group_4: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:42:21 (9741.3 seconds)
- started_at: 2026-05-04T06:19:05.364825+00:00
- finished_at: 2026-05-04T09:01:26.709365+00:00

## Matched Denominators

- pass@1: 15 matched prompts
- pass@2: 15 matched prompts
- pass@4: 15 matched prompts
- pass@8: 15 matched prompts
- pass@16: 15 matched prompts
- pass@32: 15 matched prompts
