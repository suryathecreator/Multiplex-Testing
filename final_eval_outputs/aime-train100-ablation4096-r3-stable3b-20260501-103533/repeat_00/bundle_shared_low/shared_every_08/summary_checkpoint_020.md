# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 20
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.6500 | 0.7000 | 0.7500 | 0.7500 | 0.8000 | 0.8500 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 5672.4 | 7282.9 | 10526.2 | 16884.1 | 38889.7 | 73142.4 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=20, pass@2 prompts=20, pass@4 prompts=20, pass@8 prompts=20, pass@16 prompts=20, pass@32 prompts=20

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:12:05 (725.3 seconds)
- started_at: 2026-05-04T17:26:10.119869+00:00
- finished_at: 2026-05-04T17:38:15.429785+00:00

## Matched Denominators

- pass@1: 20 matched prompts
- pass@2: 20 matched prompts
- pass@4: 20 matched prompts
- pass@8: 20 matched prompts
- pass@16: 20 matched prompts
- pass@32: 20 matched prompts
