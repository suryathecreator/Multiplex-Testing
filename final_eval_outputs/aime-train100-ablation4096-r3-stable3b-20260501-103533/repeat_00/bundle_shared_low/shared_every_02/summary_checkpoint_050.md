# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 50
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.6200 | 0.6400 | 0.6800 | 0.7200 | 0.7800 | 0.8400 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6312.2 | 8587.5 | 17208.9 | 34362.3 | 68743.5 | 137313.6 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=50, pass@2 prompts=50, pass@4 prompts=50, pass@8 prompts=50, pass@16 prompts=50, pass@32 prompts=50

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:48:35 (10115.0 seconds)
- started_at: 2026-05-03T10:20:01.767065+00:00
- finished_at: 2026-05-03T13:08:36.729904+00:00

## Matched Denominators

- pass@1: 50 matched prompts
- pass@2: 50 matched prompts
- pass@4: 50 matched prompts
- pass@8: 50 matched prompts
- pass@16: 50 matched prompts
- pass@32: 50 matched prompts
