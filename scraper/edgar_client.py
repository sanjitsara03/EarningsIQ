# Fetches SEC EDGAR filings for a given ticker.
# Four functions: get_cik, get_10q_filings, get_10k_filings, fetch_filing.
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

SEC_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT")}
CIK_CACHE = "cik_cache.json"


def get_cik(ticker: str) -> str:
    if os.path.exists(CIK_CACHE):
        with open(CIK_CACHE) as f:
            data = json.load(f)
    else:
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS,
        )
        response.raise_for_status()
        data = response.json()
        with open(CIK_CACHE, "w") as f:
            json.dump(data, f)

    ticker = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker {ticker} not found in SEC database")


def get_10q_filings(cik: str, limit: int = 5) -> list[dict]:
    return _get_filings(cik, "10-Q", limit)


def get_10k_filings(cik: str, limit: int = 3) -> list[dict]:
    return _get_filings(cik, "10-K", limit)


def _get_filings(cik: str, form: str, limit: int) -> list[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=SEC_HEADERS)
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]

    results = []
    for i, f in enumerate(recent["form"]):
        if f == form:
            results.append({
                "accession": recent["accessionNumber"][i],
                "period": recent["reportDate"][i],
                "filed_at": recent["filingDate"][i],
                "primary_doc": recent["primaryDocument"][i],
            })
        if len(results) == limit:
            break

    return results


def fetch_filing(accession: str, cik: str, primary_doc: str) -> str:
    accession_nodash = accession.replace("-", "")
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
        f"/{accession_nodash}/{primary_doc}"
    )
    response = requests.get(doc_url, headers=SEC_HEADERS)
    response.raise_for_status()
    return response.text
