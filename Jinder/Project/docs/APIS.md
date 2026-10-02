# JSearch 
### Endpoint
**Method:** `GET`
**URL:** `https://jsearch.p.rapidapi.com/search`
### Auth -> get API key from RapidAPI, then send:
- `X-RapidAPI-Key: <RAPIDAPI_KEY>`
- `X-RapidAPI-Host: jsearch.p.rapidapi.com`
### Query params
- `query` – text search  
  For Jinder we’ll put **title + location**  
  `"data scientist in Los Angeles, CA"`
- `page` – page number (`1`, `2`, `3`, …).
- `num_pages` – how many pages to fetch in one call
- `date_posted` – freshness filter: `today`, `3days`, `week`, `month`, `all`.
- `remote_jobs_only` – `"true"` / `"false"`.
- `employment_types` – `FULLTIME`, `PARTTIME`, `CONTRACT`
### example
- `GET https://jsearch.p.rapidapi.com/search`
- `headers = { X-RapidAPI-Key, X-RapidAPI-Host }`
- `params = { query, page, num_pages, date_posted, remote_jobs_only, employment_types }`
### documentation
`https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch`

# USAJobs.gov
### Endpoint
**Method:** `GET`
**URL:** `https://data.usajobs.gov/api/search`
### Auth -> register for a USAJOBS API key
- `Host: data.usajobs.gov`
- `User-Agent: <registered_email>`
- `Authorization-Key: <USAJOBS_API_KEY>`
### Query params
- `Keyword` – job title / skills
- `PositionTitle` – optional
- `LocationName` – city/region
- `DatePosted` – int 0–60 (jobs posted within last N days)
- `RemoteIndicator` – `True` or `False`
- `ResultsPerPage` – number of results per page
- `Page` – page number
### example
- `GET https://data.usajobs.gov/api/search`
- `headers = { Host, User-Agent, Authorization-Key }`
- `params = { Keyword, PositionTitle, LocationName, DatePosted, RemoteIndicator, ResultsPerPage, Page }`
### documentation
`https://developer.usajobs.gov/api-reference/get-api-search`

# Grants.gov
### Endpoint
**Method:** `POST`
**URL:** `https://api.grants.gov/v1/api/search2`
### Auth
- not required for: search2 and fetchOpportunity
### Request fields
- `keyword` – ( `"education"`, `"climate"`, `"health"`,...)
- `rows` – number of results
- `oppStatuses` – status
  - `"posted"`
  - `"forecasted|posted"`
  - `"closed|archived"`
- `eligibilities`
### example
- `POST https://api.grants.gov/v1/api/search2`
- `headers = { "Content-Type": "application/json" }`
- `json body = { keyword, rows, oppStatuses, ... }`
## using powershell
Invoke-WebRequest -Method POST "https://api.grants.gov/v1/api/search2" `
  -Headers @{ "Content-Type"="application/json" } `
  -Body '{ "keyword": "education", "rows": 5 }' `
  -OutFile "grants_output.json"
### documentation
`https://www.grants.gov/api/api-guide`