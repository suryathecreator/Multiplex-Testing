# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 45
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.6000 | 0.6222 | 0.6667 | 0.7111 | 0.7778 | 0.8222 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6579.2 | 9038.6 | 17565.5 | 35118.4 | 69964.6 | 140521.0 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=45, pass@2 prompts=45, pass@4 prompts=45, pass@8 prompts=45, pass@16 prompts=45, pass@32 prompts=45

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:14:19 (15259.1 seconds)
- started_at: 2026-05-03T10:25:19.828356+00:00
- finished_at: 2026-05-03T14:39:38.910048+00:00

## Matched Denominators

- pass@1: 45 matched prompts
- pass@2: 45 matched prompts
- pass@4: 45 matched prompts
- pass@8: 45 matched prompts
- pass@16: 45 matched prompts
- pass@32: 45 matched prompts
