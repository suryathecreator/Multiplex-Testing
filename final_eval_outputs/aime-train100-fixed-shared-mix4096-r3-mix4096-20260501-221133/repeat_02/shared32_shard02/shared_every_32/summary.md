# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 25
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.4400 | 0.4400 | 0.5200 | 0.5200 | 0.5200 | 0.5600 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6838.1 | 9627.5 | 15038.4 | 25880.0 | 47524.4 | 92170.8 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=25, pass@2 prompts=25, pass@4 prompts=25, pass@8 prompts=25, pass@16 prompts=25, pass@32 prompts=25

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:20:34 (1234.3 seconds)
- started_at: 2026-05-02T17:04:51.801137+00:00
- finished_at: 2026-05-02T17:25:26.057441+00:00

## Matched Denominators

- pass@1: 25 matched prompts
- pass@2: 25 matched prompts
- pass@4: 25 matched prompts
- pass@8: 25 matched prompts
- pass@16: 25 matched prompts
- pass@32: 25 matched prompts
