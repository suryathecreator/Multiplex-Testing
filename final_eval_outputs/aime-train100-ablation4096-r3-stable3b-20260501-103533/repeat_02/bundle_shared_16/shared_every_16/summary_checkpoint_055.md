# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 55
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 0.5273 | 0.6000 | 0.6364 | 0.6909 | 0.7091 | 0.7455 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 6411.9 | 8636.1 | 13124.8 | 22382.4 | 40453.4 | 79993.1 |

## Detailed Matched-Per-k Summary

- shared_trace_group_16: pass@1 prompts=55, pass@2 prompts=55, pass@4 prompts=55, pass@8 prompts=55, pass@16 prompts=55, pass@32 prompts=55

## Retry Statistics

- shared_trace_group_16: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:23:08 (8588.4 seconds)
- started_at: 2026-05-02T08:25:36.107096+00:00
- finished_at: 2026-05-02T10:48:44.517281+00:00

## Matched Denominators

- pass@1: 55 matched prompts
- pass@2: 55 matched prompts
- pass@4: 55 matched prompts
- pass@8: 55 matched prompts
- pass@16: 55 matched prompts
- pass@32: 55 matched prompts
