# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 70
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.5429 | 0.6000 | 0.6286 | 0.7000 | 0.7571 | 0.8143 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6584.3 | 9115.0 | 17862.0 | 35885.3 | 71081.0 | 142358.7 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=70, pass@2 prompts=70, pass@4 prompts=70, pass@8 prompts=70, pass@16 prompts=70, pass@32 prompts=70

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:14:19 (15259.1 seconds)
- started_at: 2026-05-03T10:25:19.828356+00:00
- finished_at: 2026-05-03T14:39:38.910048+00:00

## Matched Denominators

- pass@1: 70 matched prompts
- pass@2: 70 matched prompts
- pass@4: 70 matched prompts
- pass@8: 70 matched prompts
- pass@16: 70 matched prompts
- pass@32: 70 matched prompts
