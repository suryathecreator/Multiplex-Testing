# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 10
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.3000 | 0.3000 | 0.3000 | 0.5000 | 0.5000 | 0.5000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6564.9 | 9143.9 | 18416.4 | 36628.5 | 75098.0 | 148636.0 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=10, pass@2 prompts=10, pass@4 prompts=10, pass@8 prompts=10, pass@16 prompts=10, pass@32 prompts=10

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:37:39 (2259.1 seconds)
- started_at: 2026-05-02T21:04:19.272538+00:00
- finished_at: 2026-05-02T21:41:58.389777+00:00

## Matched Denominators

- pass@1: 10 matched prompts
- pass@2: 10 matched prompts
- pass@4: 10 matched prompts
- pass@8: 10 matched prompts
- pass@16: 10 matched prompts
- pass@32: 10 matched prompts
