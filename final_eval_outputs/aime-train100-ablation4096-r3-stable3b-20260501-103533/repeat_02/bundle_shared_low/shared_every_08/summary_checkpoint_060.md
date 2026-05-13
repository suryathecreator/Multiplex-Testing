# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 60
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.5167 | 0.5833 | 0.6500 | 0.6667 | 0.7333 | 0.7833 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 6326.5 | 8526.3 | 12929.4 | 21785.0 | 46292.1 | 90624.0 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=60, pass@2 prompts=60, pass@4 prompts=60, pass@8 prompts=60, pass@16 prompts=60, pass@32 prompts=60

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:12:07 (727.5 seconds)
- started_at: 2026-05-05T08:19:04.442553+00:00
- finished_at: 2026-05-05T08:31:11.932691+00:00

## Matched Denominators

- pass@1: 60 matched prompts
- pass@2: 60 matched prompts
- pass@4: 60 matched prompts
- pass@8: 60 matched prompts
- pass@16: 60 matched prompts
- pass@32: 60 matched prompts
