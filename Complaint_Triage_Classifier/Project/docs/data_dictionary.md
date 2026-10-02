# Data Dictionary

## Raw CFPB columns expected by the project

| Column | Description | Used for |
|---|---|---|
| `Date received` | Date CFPB received the complaint | Metadata, time analysis |
| `Product` | Product category selected by consumer | Weak labels, analysis |
| `Sub-product` | More specific product category | Weak labels, analysis |
| `Issue` | Issue category selected by consumer | Weak labels, analysis |
| `Sub-issue` | More specific issue category | Weak labels, analysis |
| `Consumer complaint narrative` | Consumer-submitted description of what happened | Main model input |
| `Company public response` | Optional public-facing company response | Escalation weak label, analysis |
| `Company` | Company named in complaint | Analysis only |
| `State` | Consumer state | Analysis only |
| `ZIP code` | ZIP code, possibly redacted/truncated | Not used for modeling by default |
| `Tags` | Tags such as Older American or Servicemember | Escalation weak label |
| `Consumer consent provided?` | Whether consumer consented to narrative publication | Filtering/quality |
| `Submitted via` | Submission channel | Analysis only |
| `Date sent to company` | Date sent to company | Analysis only |
| `Company response to consumer` | Company response category | Escalation weak label |
| `Timely response?` | Whether company responded timely | Escalation weak label |
| `Consumer disputed?` | Whether consumer disputed company response | Analysis only |
| `Complaint ID` | Unique complaint identifier | Deduplication, joins |

## Derived columns

| Column | Description |
|---|---|
| `raw_text` | Original narrative copied from `Consumer complaint narrative` |
| `clean_text` | Normalized text used for modeling |
| `text_char_len` | Character length of clean text |
| `text_word_len` | Word count of clean text |
| `triage_label` | Final manually verified triage label |
| `escalation_label` | Final manually verified escalation label |
| `triage_weak_label` | Rule-based candidate triage label |
| `triage_confidence` | Rule-based confidence score |
| `escalation_weak_label` | Rule-based candidate escalation label |
| `escalation_score` | Rule-based risk score |
