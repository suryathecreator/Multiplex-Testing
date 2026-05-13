# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 70
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.5571 | 0.5857 | 0.6286 | 0.6429 | 0.6714 | 0.7429 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 6268.8 | 8432.3 | 12797.3 | 21383.0 | 46580.7 | 92741.7 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=70, pass@2 prompts=70, pass@4 prompts=70, pass@8 prompts=70, pass@16 prompts=70, pass@32 prompts=70

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:12:05 (725.3 seconds)
- started_at: 2026-05-04T17:26:10.119869+00:00
- finished_at: 2026-05-04T17:38:15.429785+00:00

## Matched Denominators

- pass@1: 70 matched prompts
- pass@2: 70 matched prompts
- pass@4: 70 matched prompts
- pass@8: 70 matched prompts
- pass@16: 70 matched prompts
- pass@32: 70 matched prompts
