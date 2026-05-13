# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 5
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.6000 | 0.6000 | 0.6000 | 0.8000 | 0.8000 | 1.0000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6262.2 | 8177.4 | 16505.0 | 34050.8 | 67881.0 | 138557.6 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=5, pass@2 prompts=5, pass@4 prompts=5, pass@8 prompts=5, pass@16 prompts=5, pass@32 prompts=5

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 1:43:45 (6225.3 seconds)
- started_at: 2026-05-02T21:09:42.192777+00:00
- finished_at: 2026-05-02T22:53:27.458970+00:00

## Matched Denominators

- pass@1: 5 matched prompts
- pass@2: 5 matched prompts
- pass@4: 5 matched prompts
- pass@8: 5 matched prompts
- pass@16: 5 matched prompts
- pass@32: 5 matched prompts
