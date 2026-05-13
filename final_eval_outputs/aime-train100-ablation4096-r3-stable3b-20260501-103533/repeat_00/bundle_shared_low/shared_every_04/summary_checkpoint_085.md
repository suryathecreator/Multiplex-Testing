# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 85
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 0.4588 | 0.5059 | 0.5765 | 0.6588 | 0.7294 | 0.7647 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 6609.7 | 9221.2 | 14016.4 | 26984.0 | 54028.7 | 108702.2 |

## Detailed Matched-Per-k Summary

- shared_trace_group_4: pass@1 prompts=85, pass@2 prompts=85, pass@4 prompts=85, pass@8 prompts=85, pass@16 prompts=85, pass@32 prompts=85

## Retry Statistics

- shared_trace_group_4: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:09:24 (18563.5 seconds)
- started_at: 2026-05-04T03:33:16.560027+00:00
- finished_at: 2026-05-04T08:42:40.083942+00:00

## Matched Denominators

- pass@1: 85 matched prompts
- pass@2: 85 matched prompts
- pass@4: 85 matched prompts
- pass@8: 85 matched prompts
- pass@16: 85 matched prompts
- pass@32: 85 matched prompts
