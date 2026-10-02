# Annotation Guidelines

These guidelines define the custom labels used by the project. They are oriented
around the operational action a complaint needs, rather than copying CFPB's
product categories.

## Required columns in the final labeled file

Reviewed rows are saved as `data/labeled/verified_labels.csv`. That file must
contain these four columns:

| Column | Description |
|---|---|
| `Complaint ID` | CFPB identifier, used to deduplicate and to join back to source rows |
| `clean_text` | Normalized narrative, the model input |
| `triage_label` | One of the six triage labels below |
| `escalation_label` | Either `escalate` or `do_not_escalate` |

Additional columns are allowed and ignored. The review queue also carries a
`review_notes` column for recording uncertain calls.

## Triage labels

### 1. `fraud_identity_risk`

Use this label when the complaint is primarily about fraud, scams, identity theft, unauthorized transactions, fraudulent accounts, account takeover, security freezes, fraud alerts, or a consumer claiming that an account/reporting item is not theirs.

Positive indicators:

- "Someone opened an account in my name."
- "This transaction was unauthorized."
- "I placed a fraud alert."
- "My identity was stolen."
- "This credit inquiry is not mine."

Borderline:

- If the complaint is mainly about the company failing to investigate alleged fraud, choose this label if the fraud/identity risk is the central issue.
- If the complaint is mainly about the process of investigation and not fraud itself, choose `documentation_processing_verification`.

### 2. `billing_payment_dispute`

Use this label when the main issue is a charge, refund, payment, fee, balance, payment posting, escrow, duplicate billing, late fee, or funds not available when promised.

Positive indicators:

- "I was charged twice."
- "My payment was not applied."
- "They added fees I did not expect."
- "My refund was not issued."
- "The balance is incorrect."

Borderline:

- If the complaint involves a payment problem plus poor service, choose this label if money movement or billing is the central harm.
- If the complaint involves alleged unauthorized charges, use `fraud_identity_risk` when fraud/unauthorized activity is central.

### 3. `account_access_management`

Use this label when the main issue is opening, closing, accessing, managing, or using an account, report, card, loan, or wallet.

Positive indicators:

- "I cannot log in."
- "My account was closed without explanation."
- "I cannot access my credit report."
- "The card does not work."
- "I cannot open the account."

Borderline:

- If account access is blocked due to suspected fraud, choose `fraud_identity_risk` if fraud is central.
- If the issue is missing documentation required to open or close the account, choose `documentation_processing_verification`.

### 4. `documentation_processing_verification`

Use this label for paperwork, document validation, identity verification, evidence submission, application processing, investigation processing, refinancing, loan closing, or administrative delay.

Positive indicators:

- "I submitted documents multiple times."
- "They failed to verify the debt."
- "The investigation was incomplete."
- "My mortgage application was delayed."
- "They asked for the same paperwork again."

Borderline:

- If the issue is explicitly legal or regulatory, choose `legal_compliance_regulatory`.
- If the issue is mainly poor communication about documents, choose this label if the document/process problem is central.

### 5. `customer_service_failure`

Use this label when the main problem is poor service, lack of response, rude treatment, repeated transfers, no follow-up, call-center issues, or communication breakdown.

Positive indicators:

- "No one called me back."
- "The representative was rude."
- "I was transferred repeatedly."
- "They ignored my complaint."
- "Customer service gave conflicting answers."

Borderline:

- If poor service is secondary to billing, fraud, account access, legal, or documentation harm, choose the more specific harm category.
- Use this label when the complaint is mostly about the service experience itself.

### 6. `legal_compliance_regulatory`

Use this label when the complaint raises legal threats, regulatory violations, litigation, collection practices, foreclosure, improper contact, improper sharing, credit reporting rights, or potentially illegal actions.

Positive indicators:

- "They threatened legal action."
- "They contacted my employer."
- "This violates the FCRA."
- "They are foreclosing."
- "They sued me for a debt I do not owe."

Borderline:

- If the complaint mentions a lawyer only as context but is mainly about billing, choose `billing_payment_dispute`.
- If the complaint is about identity theft and also mentions legal rights, choose `fraud_identity_risk` if identity harm is central.

## Escalation label

### `escalate`

Use when the complaint should receive urgent specialist review because it suggests one or more of:

- fraud or identity theft,
- unauthorized activity,
- legal action or litigation,
- foreclosure or eviction risk,
- harassment or threats,
- possible regulatory violation,
- vulnerable consumer tags such as older adult or servicemember,
- untimely company response,
- severe financial harm,
- urgent access or account lockout with high impact.

### `do_not_escalate`

Use when the complaint can be routed normally and does not show immediate risk requiring urgent specialist review.

## Tie-break rules

1. Choose the label representing the main operational action needed.
2. Prefer the most specific harm category over `customer_service_failure`.
3. Use `legal_compliance_regulatory` only when legal/compliance risk is central, not merely mentioned.
4. Use `fraud_identity_risk` when unauthorized activity or identity misuse is central.
5. Use `documentation_processing_verification` when the main next step is reviewing documents, evidence, application status, or investigation process.
6. When uncertain, mark `review_notes` and keep the row in the manual review queue.

## Review volume

Aim for at least 5,000 verified rows; 8,000 to 12,000 gives the rarer categories
enough support to evaluate per-class performance meaningfully.

Review strategy, in priority order:

1. Review all low-confidence weak labels.
2. Review a balanced sample from every label.
3. Review all weakly escalated complaints.
4. Review all likely false conflicts, such as fraud vs. billing or legal vs. customer service.
