# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 90
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 0.5333 | 0.5556 | 0.6111 | 0.6778 | 0.7222 | 0.7667 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_2 | 6403.4 | 8638.5 | 17620.0 | 35243.9 | 71022.6 | 143213.1 |

## Detailed Matched-Per-k Summary

- shared_trace_group_2: pass@1 prompts=90, pass@2 prompts=90, pass@4 prompts=90, pass@8 prompts=90, pass@16 prompts=90, pass@32 prompts=90

## Retry Statistics

- shared_trace_group_2: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 4:06:20 (14780.4 seconds)
- started_at: 2026-05-03T19:15:46.434006+00:00
- finished_at: 2026-05-03T23:22:06.826025+00:00

## Matched Denominators

- pass@1: 90 matched prompts
- pass@2: 90 matched prompts
- pass@4: 90 matched prompts
- pass@8: 90 matched prompts
- pass@16: 90 matched prompts
- pass@32: 90 matched prompts
