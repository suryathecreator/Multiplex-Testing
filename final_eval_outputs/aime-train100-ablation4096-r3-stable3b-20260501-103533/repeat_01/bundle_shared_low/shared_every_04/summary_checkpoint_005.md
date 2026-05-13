# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 5
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 5575.0 | 7395.4 | 10567.2 | 18179.0 | 36383.6 | 72361.4 |

## Detailed Matched-Per-k Summary

- shared_trace_group_4: pass@1 prompts=5, pass@2 prompts=5, pass@4 prompts=5, pass@8 prompts=5, pass@16 prompts=5, pass@32 prompts=5

## Retry Statistics

- shared_trace_group_4: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:42:21 (9741.3 seconds)
- started_at: 2026-05-04T06:19:05.364825+00:00
- finished_at: 2026-05-04T09:01:26.709365+00:00

## Matched Denominators

- pass@1: 5 matched prompts
- pass@2: 5 matched prompts
- pass@4: 5 matched prompts
- pass@8: 5 matched prompts
- pass@16: 5 matched prompts
- pass@32: 5 matched prompts
