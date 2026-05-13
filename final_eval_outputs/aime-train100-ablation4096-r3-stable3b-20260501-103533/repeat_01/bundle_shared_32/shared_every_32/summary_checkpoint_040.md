# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 40
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.6000 | 0.6500 | 0.6750 | 0.7000 | 0.7000 | 0.7500 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 5988.5 | 8007.2 | 12231.2 | 20157.5 | 35948.1 | 67493.9 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=40, pass@2 prompts=40, pass@4 prompts=40, pass@8 prompts=40, pass@16 prompts=40, pass@32 prompts=40

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 1:12:09 (4329.2 seconds)
- started_at: 2026-05-02T01:24:15.235665+00:00
- finished_at: 2026-05-02T02:36:24.449743+00:00

## Matched Denominators

- pass@1: 40 matched prompts
- pass@2: 40 matched prompts
- pass@4: 40 matched prompts
- pass@8: 40 matched prompts
- pass@16: 40 matched prompts
- pass@32: 40 matched prompts
