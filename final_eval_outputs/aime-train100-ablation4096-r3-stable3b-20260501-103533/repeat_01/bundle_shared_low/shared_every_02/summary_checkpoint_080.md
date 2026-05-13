# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 80
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.5375 | 0.5875 | 0.6125 | 0.6750 | 0.7250 | 0.7750 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6598.1 | 9126.3 | 17946.4 | 35897.0 | 71374.2 | 142979.6 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=80, pass@2 prompts=80, pass@4 prompts=80, pass@8 prompts=80, pass@16 prompts=80, pass@32 prompts=80

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:14:19 (15259.1 seconds)
- started_at: 2026-05-03T10:25:19.828356+00:00
- finished_at: 2026-05-03T14:39:38.910048+00:00

## Matched Denominators

- pass@1: 80 matched prompts
- pass@2: 80 matched prompts
- pass@4: 80 matched prompts
- pass@8: 80 matched prompts
- pass@16: 80 matched prompts
- pass@32: 80 matched prompts
