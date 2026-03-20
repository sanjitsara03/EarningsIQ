import sys
sys.stdout.reconfigure(encoding='utf-8')

from edgar import set_identity, Company

set_identity("John Doe john@doe.com")


def fetch_10q(ticker: str) -> str:
    company = Company(ticker)
    filing = company.get_filings(form="10-Q").latest()
    print(f"Found 10-Q filed {filing.filing_date} — accession: {filing.accession_no}")
    return filing.text()


if __name__ == "__main__":
    text = fetch_10q("AAPL")
    print(f"\nFetched {len(text):,} characters")
    print(text[:500])
