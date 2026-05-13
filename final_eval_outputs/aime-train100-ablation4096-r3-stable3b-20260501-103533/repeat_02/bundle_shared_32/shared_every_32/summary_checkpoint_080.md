# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 80
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.5125 | 0.5750 | 0.5875 | 0.6375 | 0.6375 | 0.6500 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6671.9 | 9244.6 | 14219.1 | 24229.0 | 44243.2 | 84173.2 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=80, pass@2 prompts=80, pass@4 prompts=80, pass@8 prompts=80, pass@16 prompts=80, pass@32 prompts=80

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:48:10 (2890.0 seconds)
- started_at: 2026-05-02T07:02:14.976121+00:00
- finished_at: 2026-05-02T07:50:25.024598+00:00

## Matched Denominators

- pass@1: 80 matched prompts
- pass@2: 80 matched prompts
- pass@4: 80 matched prompts
- pass@8: 80 matched prompts
- pass@16: 80 matched prompts
- pass@32: 80 matched prompts
