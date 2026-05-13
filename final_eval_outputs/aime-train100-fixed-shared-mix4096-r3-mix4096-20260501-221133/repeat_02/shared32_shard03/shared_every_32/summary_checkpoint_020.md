# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 20
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.4000 | 0.4500 | 0.5000 | 0.5000 | 0.5000 | 0.6500 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6799.9 | 9477.6 | 14382.5 | 24738.0 | 45428.8 | 85492.8 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=20, pass@2 prompts=20, pass@4 prompts=20, pass@8 prompts=20, pass@16 prompts=20, pass@32 prompts=20

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:32:40 (1960.0 seconds)
- started_at: 2026-05-02T17:01:26.703707+00:00
- finished_at: 2026-05-02T17:34:06.709760+00:00

## Matched Denominators

- pass@1: 20 matched prompts
- pass@2: 20 matched prompts
- pass@4: 20 matched prompts
- pass@8: 20 matched prompts
- pass@16: 20 matched prompts
- pass@32: 20 matched prompts
