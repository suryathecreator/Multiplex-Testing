# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 90
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 0.4778 | 0.5444 | 0.5778 | 0.6222 | 0.6333 | 0.6889 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 6508.3 | 8842.0 | 13542.0 | 22975.5 | 41647.0 | 82025.4 |

## Detailed Matched-Per-k Summary

- shared_trace_group_16: pass@1 prompts=90, pass@2 prompts=90, pass@4 prompts=90, pass@8 prompts=90, pass@16 prompts=90, pass@32 prompts=90

## Retry Statistics

- shared_trace_group_16: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:23:08 (8588.4 seconds)
- started_at: 2026-05-02T08:25:36.107096+00:00
- finished_at: 2026-05-02T10:48:44.517281+00:00

## Matched Denominators

- pass@1: 90 matched prompts
- pass@2: 90 matched prompts
- pass@4: 90 matched prompts
- pass@8: 90 matched prompts
- pass@16: 90 matched prompts
- pass@32: 90 matched prompts
