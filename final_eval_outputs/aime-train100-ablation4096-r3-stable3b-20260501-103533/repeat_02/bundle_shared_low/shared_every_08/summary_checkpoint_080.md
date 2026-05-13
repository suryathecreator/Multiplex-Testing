# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 80
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 0.5125 | 0.5625 | 0.6125 | 0.6250 | 0.7000 | 0.7625 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_8 | 6383.9 | 8660.3 | 13093.4 | 22223.4 | 46591.4 | 92529.0 |

## Detailed Matched-Per-k Summary

- shared_trace_group_8: pass@1 prompts=80, pass@2 prompts=80, pass@4 prompts=80, pass@8 prompts=80, pass@16 prompts=80, pass@32 prompts=80

## Retry Statistics

- shared_trace_group_8: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:12:07 (727.5 seconds)
- started_at: 2026-05-05T08:19:04.442553+00:00
- finished_at: 2026-05-05T08:31:11.932691+00:00

## Matched Denominators

- pass@1: 80 matched prompts
- pass@2: 80 matched prompts
- pass@4: 80 matched prompts
- pass@8: 80 matched prompts
- pass@16: 80 matched prompts
- pass@32: 80 matched prompts
