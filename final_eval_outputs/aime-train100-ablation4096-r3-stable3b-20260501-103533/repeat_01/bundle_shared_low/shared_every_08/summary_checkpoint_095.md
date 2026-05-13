# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 95
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.4842 | 0.5368 | 0.5895 | 0.6316 | 0.7263 | 0.7789 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 6483.9 | 8804.0 | 13581.6 | 23324.7 | 46264.9 | 94087.1 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=95, pass@2 prompts=95, pass@4 prompts=95, pass@8 prompts=95, pass@16 prompts=95, pass@32 prompts=95

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:12:06 (18725.9 seconds)
- started_at: 2026-05-04T12:31:04.620487+00:00
- finished_at: 2026-05-04T17:43:10.518641+00:00

## Matched Denominators

- pass@1: 95 matched prompts
- pass@2: 95 matched prompts
- pass@4: 95 matched prompts
- pass@8: 95 matched prompts
- pass@16: 95 matched prompts
- pass@32: 95 matched prompts
