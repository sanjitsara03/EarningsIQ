from scraper.edgar_client import get_cik, get_10q_filings, fetch_filing

cik = get_cik('AAPL')
print(f'CIK: {cik}')

filings = get_10q_filings(cik, limit=2)
print(f'10-Q filings: {filings}')

f = filings[0]
text = fetch_filing(f['accession'], cik, f['primary_doc'])
print(f'Filing length: {len(text):,} chars')
print(text[:300])