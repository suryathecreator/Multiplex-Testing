# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 75
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.5600 | 0.5867 | 0.6267 | 0.6400 | 0.6667 | 0.7333 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 6310.4 | 8503.2 | 12947.2 | 21690.1 | 46816.7 | 93170.5 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=75, pass@2 prompts=75, pass@4 prompts=75, pass@8 prompts=75, pass@16 prompts=75, pass@32 prompts=75

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:12:05 (725.3 seconds)
- started_at: 2026-05-04T17:26:10.119869+00:00
- finished_at: 2026-05-04T17:38:15.429785+00:00

## Matched Denominators

- pass@1: 75 matched prompts
- pass@2: 75 matched prompts
- pass@4: 75 matched prompts
- pass@8: 75 matched prompts
- pass@16: 75 matched prompts
- pass@32: 75 matched prompts
