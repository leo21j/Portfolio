import json
import logging
import os
import re
from pathlib import Path

import fitz

from services.job_matcher.src.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class JobMatcher:
    def __init__(
        self,
        alpha: float = 0.5,
        use_graph: bool = True,
        use_rerank: bool = True
    ):
        self.retriever = HybridRetriever(
            alpha=alpha,
            use_rerank=use_rerank,
            use_graph=use_graph
        )
        self.jobs_metadata = {}
        self.normalized_jobs = []

        logger.info(f"Job Matcher initialized (alpha={alpha}, graph={use_graph})")


    """
    Parsing jobs from metadata file and apply normalization,
    then using the HybridRetriever to encode job chunks to embeddings.
    """
    def load_jobs_from_json(self, json_file: str):
        """
        Supporting three different json files located in sample_data
        """
        logger.info(f"Loading jobs from {json_file}...")

        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"Jobs file not found: {json_file}")

        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)

        # normalized_jobs = []

        # Handle different JSON structures from sample_data
        if isinstance(data, dict):
            # JSearch
            if "data" in data and isinstance(data["data"], list):
                request_id = data.get("request_id")
                for job in data["data"]:
                    if not isinstance(job, dict):
                        continue
                    j = dict(job)
                    j["_source"] = "JSEARCH"
                    j["_jsearch_id"] = j.get("job_id")
                    if request_id:
                        j["_jsearch_request_id"] = request_id

                    self.normalized_jobs.append(j)

            # USAJOBS
            elif 'SearchResult' in data and isinstance(data['SearchResult'], dict):
                items = data["SearchResult"].get("SearchResultItems", [])
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    desc = item.get("MatchedObjectDescriptor") or {}
                    if not isinstance(desc, dict):
                        continue

                    j = dict(desc)
                    j["_source"] = "USAJOBS"
                    j["_usajobs_id"] = item.get("MatchedObjectId")

                    self.normalized_jobs.append(j)

            elif "data" in data and isinstance(data["data"], dict) and "oppHits" in data["data"]:
                for item in data["data"]["oppHits"]:
                    if not isinstance(item, dict):
                        continue

                    j = {
                        "job_id": item.get("id") or item.get("number"),
                        "job_title": item.get("title"),
                        "employer_name": item.get("agency"),
                        "job_description": (
                            f"Funding opportunity {item.get('number')} from {item.get('agency')} "
                            f"(status: {item.get('oppStatus')}, type: {item.get('docType')}). "
                            f"CFDA: {', '.join(item.get('cfdaList', []))}"
                        ),
                        "_source": "GRANTS",
                        "_grants_id": item.get("id"),
                    }

                    self.normalized_jobs.append(j)

            else:
                j = dict(data)
                j.setdefault("_source", "SINGLE_DICT")
                self.normalized_jobs.append(j)

        elif isinstance(data, list):
            for job in data:
                if not isinstance(job, dict):
                    continue
                j = dict(job)
                j.setdefault("_source", "LIST")
                self.normalized_jobs.append(j)
        else:
            raise ValueError("Invalid JSON format")

        logger.info(f"Loaded {len(self.normalized_jobs)} jobs from {json_file}")

    def load_jobs_from_dict(self, data: dict, source: str):
        """
        this will be the one called on aws
        """
        if isinstance(data, dict):
            # JSearch
            if "data" in data and isinstance(data["data"], list):
                request_id = data.get("request_id")
                for job in data["data"]:
                    if not isinstance(job, dict):
                        continue
                    j = dict(job)
                    j["_source"] = "JSEARCH"
                    j["_jsearch_id"] = j.get("job_id")
                    if request_id:
                        j["_jsearch_request_id"] = request_id
                    self.normalized_jobs.append(j)

            # USAJOBS
            elif 'SearchResult' in data and isinstance(data['SearchResult'], dict):
                items = data["SearchResult"].get("SearchResultItems", [])
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    desc = item.get("MatchedObjectDescriptor") or {}
                    if not isinstance(desc, dict):
                        continue
                    j = dict(desc)
                    j["_source"] = "USAJOBS"
                    j["_usajobs_id"] = item.get("MatchedObjectId")
                    self.normalized_jobs.append(j)

            # Grants
            elif "data" in data and isinstance(data["data"], dict) and "oppHits" in data["data"]:
                for item in data["data"]["oppHits"]:
                    if not isinstance(item, dict):
                        continue
                    j = {
                        "job_id": item.get("id") or item.get("number"),
                        "job_title": item.get("title"),
                        "employer_name": item.get("agency"),
                        "job_description": (
                            f"Funding opportunity {item.get('number')} from {item.get('agency')} "
                            f"(status: {item.get('oppStatus')}, type: {item.get('docType')}). "
                            f"CFDA: {', '.join(item.get('cfdaList', []))}"
                        ),
                        "_source": "GRANTS",
                        "_grants_id": item.get("id"),
                    }
                    self.normalized_jobs.append(j)

        logger.info(f"Loaded {len(self.normalized_jobs)} total jobs from {source}")

    def load_jobs(self, jobs: list[dict]):
        """
        Get jobs info, from the json files
        """
        if not jobs:
            raise ValueError("No jobs provided")

        logger.info(f"Processing {len(jobs)} jobs...")

        chunks = []
        for i, job in enumerate(jobs):
            source_tag = job.get("_source")

            job_id = (
                    job.get("job_id")
                    or job.get("_jsearch_id")
                    or job.get("_usajobs_id")
                    or job.get("_grants_id")
                    # fallback
                    or f"{source_tag or 'job'}_{i}"
            )


            job_text = self._extract_job_text(job)

            employer = job.get("employer_name") or job.get("OrganizationName")

            if employer and source_tag:
                chunk_source = f"{employer} ({source_tag})"
            elif employer:
                chunk_source = employer
            elif source_tag:
                chunk_source = source_tag
            else:
                chunk_source = "Unknown Company"

            chunk = {
                "chunk_id": job_id,
                "text": job_text,
                "source": chunk_source,
            }

            chunks.append(chunk)

            self.jobs_metadata[job_id] = job

        #  FAISS indexing
        index_dir = Path("data")
        index_dir.mkdir(exist_ok=True)

        index_file = "sample_data/fassi_index/jobs_index.faiss"
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        self.retriever.build_index(chunks, index_file=index_file)

        logger.info(f"Successfully indexed {len(chunks)} jobs")

    def _extract_job_text(self, job: dict) -> str:
        """
        Extract from job posting.
        """
        j = []

        # Title
        title = job.get("job_title") or job.get("PositionTitle", "")
        if title:
            j.append(title)
            j.append(title)

        # Description
        description = job.get("job_description", "")

        if not description:
            qual_summary = job.get("QualificationSummary", "")
            user_area = job.get("UserArea") or {}
            if isinstance(user_area, dict):
                details = user_area.get("Details") or {}
            else:
                details = {}
            job_summary = details.get("JobSummary", "")

            description = qual_summary or job_summary

        if description:
            j.append(description[:2500])

        # Highlights (JSearch only)
        highlights = job.get("job_highlights", {})

        quals = highlights.get("Qualifications", [])
        if quals:
            j.append("Required qualifications: " + " ".join(quals[:15]))

        resp = highlights.get("Responsibilities", [])
        if resp:
            j.append("Responsibilities: " + " ".join(resp[:10]))

        benefits = highlights.get("Benefits", [])
        if benefits:
            j.append("Benefits: " + " ".join(benefits[:5]))

        # Location
        city = job.get("job_city", "")
        state = job.get("job_state", "")

        if not city and not state:
            locs = job.get("PositionLocation")
            if isinstance(locs, list) and locs:
                loc = locs[0]
                if isinstance(loc, dict):
                    city = loc.get("CityName") or loc.get("LocationName") or ""
                    state = loc.get("CountrySubDivisionCode") or ""

        if city or state:
            j.append(f"Location: {city}, {state}")

        return " ".join(j)

    def generate_embeddings(self):
        self.load_jobs(self.normalized_jobs)

    """
    Parsing user's resume from pdf to a normalized dictionary.
    """
    def parse_from_resume(self, path: str):
        """
        Main function that will parse resume from pdf file to dict data
        """
        try:
            text = self.load_resume_from_pdf(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load resume file {path}: {e}") from e

        # Only using full_text for now, the other parts stay for future possible improvements
        resume_info = {
            "full_text": text,
            "education": self.get_info_by_resume_section(text, "EDUCATION"),
            "experience": self.get_info_by_resume_section(text, "EXPERIENCE"),
            "project": self.get_info_by_resume_section(text, "PROJECTS"),
            "skills": self.get_info_by_resume_section(text, "SKILLS"),
        }

        return resume_info

    def load_resume_from_pdf(self, pdf_path: str):
        """
        Loading text from resume (has to be a .pdf)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file {pdf_path} does not exist")

        text = ""

        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text("text")
                text += page_text + "\n\n"
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF file {pdf_path}: {e}") from e

        return text.strip()

    def get_info_by_resume_section(self, text: str, keyword: str):
        """
        Get infomation by resume section.
        """
        if not text:
            raise ValueError("No text provided")

        pattern = rf"\n\s*{re.escape(keyword)}\s*:?\s*\n"
        match = re.search(pattern, text)

        if not match:
            return ""

        start_idx = match.end()

        resume_rest_part = text[start_idx:]

        next_section = re.search(r'\n\s*[A-Z][A-Za-z\s]{3,20}:?\s*\n', resume_rest_part)

        if next_section:
            end_idx = next_section.start()
            return resume_rest_part[:end_idx].strip()
        else:
            return resume_rest_part[:800].strip()

    """
    Matching part below...
    Due to the time limits, here I will only implement the very basic method,
    which will be using the whole resume text as a Query, and use this to search
    in the pre-generated job embedding space.
    """
    def match_job(self, resume_text: str, top_k: int = 10):

        if not resume_text or not resume_text.strip():
            raise ValueError("Empty resume text")

        results = self.retriever.search(
            query=resume_text,
            k=top_k
        )

        matches = []

        for chunk, score in results:
            job_id = chunk["chunk_id"]
            job = self.jobs_metadata.get(job_id)

            title = job.get("job_title") or job.get("PositionTitle")
            company = job.get("employer_name") or job.get("OrganizationName")

            city = job.get("job_city", "")
            state = job.get("job_state", "")
            location = ", ".join([p for p in [city, state] if p])

            matches.append({
                "job_id": job_id,
                "title": title,
                "company": company,
                "location": location,
                "match_score": float(score),
            })

        return matches
