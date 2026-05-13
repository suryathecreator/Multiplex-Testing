# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 60
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.5833 | 0.6000 | 0.6167 | 0.7167 | 0.7667 | 0.8000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6360.9 | 8532.5 | 17305.0 | 34664.3 | 69999.0 | 140560.6 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=60, pass@2 prompts=60, pass@4 prompts=60, pass@8 prompts=60, pass@16 prompts=60, pass@32 prompts=60

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:06:20 (14780.4 seconds)
- started_at: 2026-05-03T19:15:46.434006+00:00
- finished_at: 2026-05-03T23:22:06.826025+00:00

## Matched Denominators

- pass@1: 60 matched prompts
- pass@2: 60 matched prompts
- pass@4: 60 matched prompts
- pass@8: 60 matched prompts
- pass@16: 60 matched prompts
- pass@32: 60 matched prompts
