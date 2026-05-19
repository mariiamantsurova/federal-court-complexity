# Event Log.csv — schema reference

**Path**: `Event Log.csv` (~2.7 GB, 4,811,483 rows, 98 columns)

## Keys & time

| Column | Role |
|--------|------|
| (index) | Row id |
| `ucid` | Case id (e.g. `ilnd;;1:00-cr-00238`) — **group key** |
| `date_filed` | Event date (YYYY-MM-DD) |
| `Activity` | Broad event type (26 types in SCALES ontology) |
| `event_judge` | Judge tuple string; parse to id + role |

## Activity attributes (61 columns)

Boolean flags `attribute_*` (stored as True/False strings). Examples:

- Process: `attribute_scheduling`, `attribute_rescheduling`, `attribute_hearing_conf`
- Motions: `attribute_motion_to_dismiss`, `attribute_motion_for_summary_judgment`, …
- Outcomes: `attribute_dispositive`, `attribute_granting_motion_to_dismiss`, `attribute_settlement_consent_decree`, `attribute_voluntary_dismissal`, …
- Notices, stipulations, transfers, trial types, pleas (criminal), etc.

`attribute_duplicates` — integer count of duplicate filings for same event (usually 1).

## Case metadata (row-level, repeated per event)

| Column | Notes |
|--------|-------|
| `city` | Venue (e.g. Chicago) |
| `case_status` | open / closed |
| `case_type` | cv / cr |
| `nature_suit` | Civil nature code + label; "Not Applicable" for cr |
| `is_mdl` | MDL flag (rare) |
| `District_Judge`, `Magistrate_Judge` | Judge ids |
| `plaintiffs_*`, `Defendants_*` | Counts and shares (ind, pro_se, counsels) |
| `Defendants_highest_offense_*`, `*_counts` | Criminal fields |
| `Party_*` | Binary party-role flags |
| `Other_courts`, `related_case_count` | Related litigation |

## Activity frequency (full file)

minute_entry, motion, notice, order, response, complaint, summons, answer, settlement, judgment, stipulation, waiver, …

## Engineering notes

- **Grain**: one row = one event; case features require `groupby('ucid')`.
- **Sparsity**: most `attribute_*` are False >99% of rows.
- **Imbalance**: ~closed >> open at event-row level because closed cases contribute more events.
- **cv vs cr**: different `nature_suit` semantics; separate pipelines recommended.

## SCALES docs

https://docs.scales-okn.org/eventlog/
