# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 15
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.4667 | 0.4667 | 0.4667 | 0.4667 | 0.5333 | 0.5333 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6975.5 | 9843.0 | 15662.5 | 27227.3 | 49305.5 | 94186.6 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=15, pass@2 prompts=15, pass@4 prompts=15, pass@8 prompts=15, pass@16 prompts=15, pass@32 prompts=15

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:34:59 (2099.0 seconds)
- started_at: 2026-05-02T12:14:25.184318+00:00
- finished_at: 2026-05-02T12:49:24.172259+00:00

## Matched Denominators

- pass@1: 15 matched prompts
- pass@2: 15 matched prompts
- pass@4: 15 matched prompts
- pass@8: 15 matched prompts
- pass@16: 15 matched prompts
- pass@32: 15 matched prompts
