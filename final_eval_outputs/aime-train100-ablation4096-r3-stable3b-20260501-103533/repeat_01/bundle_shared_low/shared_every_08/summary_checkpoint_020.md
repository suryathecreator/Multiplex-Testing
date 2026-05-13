# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 20
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.8000 | 0.8000 | 0.8000 | 0.8500 | 0.9000 | 0.9000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 5717.6 | 7322.8 | 10494.6 | 17437.7 | 34045.6 | 69055.9 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=20, pass@2 prompts=20, pass@4 prompts=20, pass@8 prompts=20, pass@16 prompts=20, pass@32 prompts=20

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:12:06 (18725.9 seconds)
- started_at: 2026-05-04T12:31:04.620487+00:00
- finished_at: 2026-05-04T17:43:10.518641+00:00

## Matched Denominators

- pass@1: 20 matched prompts
- pass@2: 20 matched prompts
- pass@4: 20 matched prompts
- pass@8: 20 matched prompts
- pass@16: 20 matched prompts
- pass@32: 20 matched prompts
