from typing import List, Optional

from google_play_scraper import search, app as gp_app
from rapidfuzz import process, fuzz

from .models import PackageCandidate


class PackageIdentifierFinder:
    def __init__(self, lang: str = "en", country: str = "us") -> None:
        self.lang = lang
        self.country = country

    def search_candidates(
        self, query: str, max_results: int = 10
    ) -> List[PackageCandidate]:
        results = search(
            query, lang=self.lang, country=self.country, n_hits=max_results
        )
        candidates: List[PackageCandidate] = []
        for r in results:
            app_id = r.get("appId", "")
            if not app_id:
                continue
            title = r.get("title", "")
            developer = r.get("developer", "")
            score = float(r.get("score", 0.0))
            installs = r.get("installs", "")
            icon = r.get("icon", "")
            url = (
                f"https://play.google.com/store/apps/details?id={app_id}"
                if app_id
                else ""
            )
            candidates.append(
                PackageCandidate(
                    app_id=app_id,
                    title=title,
                    developer=developer,
                    score=score,
                    installs=installs,
                    icon=icon,
                    url=url,
                )
            )
        return candidates

    def best_match(
        self, query: str, max_results: int = 15
    ) -> Optional[PackageCandidate]:
        # direct lookup if query is likely an appId
        if "." in query and query.count(".") >= 1 and query.strip().islower():
            try:
                details = gp_app(query.strip(), lang=self.lang, country=self.country)
                return PackageCandidate(
                    app_id=query.strip(),
                    title=details.get("title", query.strip()),
                    developer=details.get("developer", ""),
                    score=float(details.get("score", 0.0) or 0.0),
                    installs=details.get("installs", ""),
                    icon=details.get("icon", ""),
                    url=f"https://play.google.com/store/apps/details?id={query.strip()}",
                )
            except Exception:
                pass

        candidates = self.search_candidates(query, max_results=max_results)
        if not candidates:
            return None

        choices = [
            (c.title + " " + c.developer, idx) for idx, c in enumerate(candidates)
        ]
        match = process.extractOne(query, [c[0] for c in choices], scorer=fuzz.WRatio)
        if not match:
            return candidates[0]

        _, _, best_idx = match
        return candidates[best_idx]
