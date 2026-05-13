# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 100
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.5200 | 0.5300 | 0.5800 | 0.6500 | 0.6900 | 0.7400 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6479.8 | 9014.0 | 18025.3 | 36149.9 | 72641.3 | 144584.1 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=100, pass@2 prompts=100, pass@4 prompts=100, pass@8 prompts=100, pass@16 prompts=100, pass@32 prompts=100

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:48:35 (10115.0 seconds)
- started_at: 2026-05-03T10:20:01.767065+00:00
- finished_at: 2026-05-03T13:08:36.729904+00:00

## Matched Denominators

- pass@1: 100 matched prompts
- pass@2: 100 matched prompts
- pass@4: 100 matched prompts
- pass@8: 100 matched prompts
- pass@16: 100 matched prompts
- pass@32: 100 matched prompts
