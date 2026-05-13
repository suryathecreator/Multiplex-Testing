# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 10
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6567.7 | 9079.0 | 14072.4 | 23371.1 | 43033.4 | 81490.9 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=10, pass@2 prompts=10, pass@4 prompts=10, pass@8 prompts=10, pass@16 prompts=10, pass@32 prompts=10

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:35:22 (2122.3 seconds)
- started_at: 2026-05-02T13:12:10.854422+00:00
- finished_at: 2026-05-02T13:47:33.202590+00:00

## Matched Denominators

- pass@1: 10 matched prompts
- pass@2: 10 matched prompts
- pass@4: 10 matched prompts
- pass@8: 10 matched prompts
- pass@16: 10 matched prompts
- pass@32: 10 matched prompts
