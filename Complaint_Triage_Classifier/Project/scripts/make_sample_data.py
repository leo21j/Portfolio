"""Generate the synthetic sample dataset used for smoke-testing the pipeline.

Every narrative in this file was written by hand for this fixture. None of the
text, company names, or identifiers come from a real complaint. ZIP codes are
always masked and the Complaint IDs are sequential fakes.

The output is committed as ``data/raw/complaints_sample.csv`` so the pipeline
runs on a fresh clone without downloading the real CFPB export. It is far too
small and too clean to train a usable model -- see the README for how to get
the real data.

Usage:
    python scripts/make_sample_data.py
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# CFPB export column order, reproduced so the fixture validates against
# complaint_triage.constants.RAW_REQUIRED_COLUMNS.
COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Consumer consent provided?",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID",
]

# Five hand-written narratives per triage category.
NARRATIVES: dict[str, list[str]] = {
    "fraud_identity_risk": [
        "Someone opened an account in my name without my permission and I never applied "
        "for it. This is identity theft and I want the account removed.",
        "I found an unauthorized transaction on my statement for a purchase I did not "
        "make. I believe my card was stolen and used fraudulently.",
        "A fake account appeared on my credit report. This account is not mine and I "
        "have placed a fraud alert with all three bureaus.",
        "My online profile was hacked and the attacker changed my mailing address. I "
        "reported the account takeover but nothing has been corrected.",
        "There is a fraudulent inquiry on my report from a company I never contacted. "
        "I am asking for a security freeze on my file.",
    ],
    "billing_payment_dispute": [
        "I was charged twice for the same monthly payment and the duplicate charge has "
        "not been refunded after several weeks.",
        "They added a late fee even though my payment was submitted before the due "
        "date. The balance shown on my statement is incorrect.",
        "My payment was not applied to the correct loan and now the escrow account "
        "shows a shortage that I do not owe.",
        "I was overcharged interest on my statement and the refund I was promised was "
        "never issued to my account.",
        "I dispute the fees on my latest statement. The amount charged does not match "
        "the terms I agreed to when I opened the account.",
    ],
    "account_access_management": [
        "I cannot log in to my online account. The password reset link never arrives "
        "and I am locked out of my own records.",
        "My account was closed without any explanation or notice, and I am unable to "
        "access the funds that remain in it.",
        "I am unable to get my credit report through the website. Every attempt to "
        "access my credit score fails with an error.",
        "The replacement card they sent is not working at any terminal. I have been "
        "unable to use the card since it arrived.",
        "I tried to open an account and the application was blocked with no reason "
        "given. Nobody can tell me how to manage this.",
    ],
    "documentation_processing_verification": [
        "I submitted the same paperwork three times and they still claim the documents "
        "were never received for verification.",
        "They failed to verify the debt after I requested validation in writing. The "
        "investigation was closed without any evidence.",
        "My mortgage application has been delayed for months. Each time I call they "
        "ask for the same documents again.",
        "The investigation into the incorrect information on my report was incomplete "
        "and no proof was ever provided to me.",
        "I sent proof of payment as part of my refinance application and the "
        "processing has stalled with no status update.",
    ],
    "customer_service_failure": [
        "No one has called me back after four messages. The representative promised a "
        "supervisor would follow up and nobody did.",
        "I was transferred repeatedly between departments and the last agent hung up "
        "on me without resolving anything.",
        "Customer service gave me three conflicting answers about the same question. "
        "There has been no response to my written complaint.",
        "The representative was rude and refused to escalate my issue. I waited on "
        "hold for over an hour before being disconnected.",
        "They ignored my complaint entirely. I have never received any written "
        "communication from a representative despite repeated requests.",
    ],
    "legal_compliance_regulatory": [
        "The collector threatened legal action and said they would take me to court "
        "over a debt that I do not owe.",
        "They contacted my employer about my account, which I believe is an improper "
        "sharing of my information.",
        "This reporting error violates the FCRA and I have asked for correction twice "
        "with no result. I may contact an attorney.",
        "The servicer started foreclosure proceedings while my loan modification "
        "request was still under review.",
        "They threatened garnishment of my wages for an account that was already "
        "discharged in bankruptcy.",
    ],
}

# CFPB product/issue values chosen to match configs/label_schema.yaml, so the
# weak labeler exercises both its keyword and metadata rules.
PRODUCTS: dict[str, tuple[str, str]] = {
    "fraud_identity_risk": (
        "Credit reporting or other personal consumer reports",
        "Credit reporting",
    ),
    "billing_payment_dispute": (
        "Credit card or prepaid card",
        "General-purpose credit card or charge card",
    ),
    "account_access_management": ("Checking or savings account", "Checking account"),
    "documentation_processing_verification": ("Mortgage", "Conventional home mortgage"),
    "customer_service_failure": ("Debt collection", "Credit card debt"),
    "legal_compliance_regulatory": ("Debt collection", "Other debt"),
}

# High-risk issues (also present in the schema's escalation rules).
ISSUES_HIGH_RISK: dict[str, tuple[str, str]] = {
    "fraud_identity_risk": ("Fraud or scam", ""),
    "billing_payment_dispute": (
        "Unauthorized transactions or other transaction problem",
        "",
    ),
    "account_access_management": ("Problem with fraud alerts or security freezes", ""),
    "documentation_processing_verification": ("Loan modification,collection,foreclosure", ""),
    "customer_service_failure": (
        "Threatened to contact someone or share information improperly",
        "",
    ),
    "legal_compliance_regulatory": ("Took or threatened to take negative or legal action", ""),
}

# Routine issues that should not trigger escalation.
ISSUES_ROUTINE: dict[str, tuple[str, str]] = {
    "fraud_identity_risk": ("Credit monitoring or identity theft protection services", ""),
    "billing_payment_dispute": ("Fees or interest", "Problem with fees"),
    "account_access_management": ("Closing an account", "Can't close your account"),
    "documentation_processing_verification": (
        "Incorrect information on your report",
        "Account information incorrect",
    ),
    "customer_service_failure": ("Communication tactics", "Frequent or repeated calls"),
    "legal_compliance_regulatory": ("Attempts to collect debt not owed", "Debt was paid"),
}

# Invented company names. Any resemblance to a real institution is accidental.
COMPANIES = [
    "NORTHWIND FINANCIAL LLC",
    "Lakeside Credit Union",
    "ATLAS SERVICING CO.",
    "Brightline Bank N.A.",
    "MERIDIAN RECOVERY GROUP",
    "Cobblestone Mortgage Inc.",
]
STATES = ["CA", "TX", "FL", "NY", "IL", "OH", "GA", "NC", "PA", "AZ", "WA", "CO"]
CHANNELS = ["Web", "Phone", "Referral", "Postal mail"]

PUBLIC_RESPONSE = (
    "Company has responded to the consumer and the CFPB and chooses not to "
    "provide a public response"
)


def build_rows(num_rows: int, seed: int) -> list[dict[str, str]]:
    """Build synthetic complaint records covering every label combination."""
    rng = random.Random(seed)
    categories = list(NARRATIVES)
    rows: list[dict[str, str]] = []
    complaint_id = 7_100_000

    for index in range(num_rows):
        category = categories[index % len(categories)]
        # Fraud and legal complaints escalate most of the time; others rarely.
        high_risk = index % len(categories) in (0, 5) or index % 7 == 0
        issue, sub_issue = (ISSUES_HIGH_RISK if high_risk else ISSUES_ROUTINE)[category]
        product, sub_product = PRODUCTS[category]
        narrative = NARRATIVES[category][(index // len(categories)) % 5]
        complaint_id += rng.randint(17, 400)

        rows.append(
            {
                "Date received": _fake_date(rng),
                "Product": product,
                "Sub-product": sub_product,
                "Issue": issue,
                "Sub-issue": sub_issue,
                "Consumer complaint narrative": narrative,
                "Company public response": PUBLIC_RESPONSE,
                "Company": COMPANIES[index % len(COMPANIES)],
                "State": STATES[index % len(STATES)],
                # Masked, matching how CFPB truncates ZIP codes for some records.
                "ZIP code": f"{rng.randint(100, 999)}XX",
                "Tags": "Older American" if high_risk and index % 5 == 0 else "",
                "Consumer consent provided?": "Consent provided",
                "Submitted via": CHANNELS[index % len(CHANNELS)],
                "Date sent to company": _fake_date(rng),
                "Company response to consumer": (
                    "Untimely response"
                    if high_risk and index % 4 == 0
                    else "Closed with explanation"
                ),
                "Timely response?": "No" if high_risk and index % 3 == 0 else "Yes",
                "Consumer disputed?": "",
                "Complaint ID": str(complaint_id),
            }
        )

    return rows


def _fake_date(rng: random.Random) -> str:
    return f"202{rng.randint(3, 5)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data/raw/complaints_sample.csv")
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(num_rows=args.rows, seed=args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows to {output}")


if __name__ == "__main__":
    main()
