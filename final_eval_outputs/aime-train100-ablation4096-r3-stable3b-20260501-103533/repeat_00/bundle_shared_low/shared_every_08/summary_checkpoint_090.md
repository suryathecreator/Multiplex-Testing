# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 90
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.5444 | 0.5889 | 0.6222 | 0.6333 | 0.6667 | 0.7222 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 6295.0 | 8466.1 | 12938.5 | 21669.2 | 46578.5 | 93637.1 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=90, pass@2 prompts=90, pass@4 prompts=90, pass@8 prompts=90, pass@16 prompts=90, pass@32 prompts=90

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:12:05 (725.3 seconds)
- started_at: 2026-05-04T17:26:10.119869+00:00
- finished_at: 2026-05-04T17:38:15.429785+00:00

## Matched Denominators

- pass@1: 90 matched prompts
- pass@2: 90 matched prompts
- pass@4: 90 matched prompts
- pass@8: 90 matched prompts
- pass@16: 90 matched prompts
- pass@32: 90 matched prompts
