import json
import unittest
from copy import deepcopy
from paper_trail.compact import compact_result


class CompactChecks(unittest.TestCase):
    def test_feed_projection_preserves_evidence_and_raw_cache(self):
        raw = {
            "ok": True,
            "cache_hit": True,
            "data": [
                {
                    "paper": {
                        "id": "2602.08025",
                        "title": "Paper",
                        "summary": "a" * 5000,
                        "publishedAt": "2026-02-08",
                        "authors": [{"name": "Author", "avatar": "x" * 10000}],
                    },
                    "publishedAt": "2026-02-09",
                    "summary": "a" * 5000,
                    "submittedBy": {
                        "name": "alice",
                        "fullname": "Alice",
                        "avatar": "x" * 10000,
                    },
                    "organization": {
                        "name": "lab",
                        "fullname": "Research Lab",
                        "avatar": "ignored",
                    },
                }
            ],
        }
        original = deepcopy(raw)
        result = compact_result("/api/daily_papers", raw)
        self.assertEqual(raw, original)
        item = result["data"][0]
        self.assertEqual(item["published_at"], "2026-02-08")
        self.assertEqual(item["feed_published_at"], "2026-02-09")
        self.assertTrue(item["abstract_truncated"])
        self.assertEqual(item["submitted_by"]["name"], "alice")
        self.assertEqual(item["organization"]["display_name"], "Research Lab")
        self.assertEqual(item["organization"]["url"], "https://huggingface.co/lab")
        self.assertEqual(len(item["abstract"]), 4001)
        self.assertTrue(result["cache_hit"])
        self.assertLess(len(json.dumps(result)), len(json.dumps(raw)) / 5)

    def test_metadata_and_resource_whitelist(self):
        result = compact_result(
            "/api/papers/2602.08025",
            {
                "ok": True,
                "data": {
                    "id": "2602.08025",
                    "title": "Paper",
                    "summary": "z" * 5000,
                    "authors": [{"name": "A", "avatar": "ignored"}] * 15,
                    "githubRepo": "https://github.com/org/repo",
                },
            },
        )
        self.assertEqual(len(result["data"]["authors"]), 12)
        self.assertEqual(result["data"]["authors_omitted"], 3)
        self.assertTrue(result["data"]["abstract_truncated"])
        result = compact_result(
            "/api/models",
            {
                "ok": True,
                "data": [{"id": "org/model", "tags": ["x"] * 10000, "downloads": 5}],
            },
        )
        self.assertNotIn("tags", result["data"][0])
        self.assertEqual(result["data"][0]["url"], "https://huggingface.co/org/model")

    def test_abstract_limit_for_lists_and_details(self):
        for length in (500, 3999, 4000, 4001):
            paper = {"id": "2602.08025", "summary": "文" * length}
            for path, data in (
                ("/api/daily_papers", [{"paper": paper}]),
                ("/api/papers/search", [paper]),
                ("/api/papers/2602.08025", paper),
            ):
                result = compact_result(path, {"ok": True, "data": data})["data"]
                item = result[0] if isinstance(result, list) else result
                self.assertEqual(
                    item["abstract"],
                    "文" * min(length, 4000) + ("…" if length > 4000 else ""),
                )
                self.assertEqual(item["abstract_truncated"], length > 4000)

    def test_ai_fields_and_author_accounts(self):
        paper = {
            "id": "2602.08025",
            "ai_summary": "AI overview",
            "ai_keywords": ["agent", "memory"],
            "authors": [
                {
                    "name": "Alice Smith",
                    "user": {"name": "alice", "avatarUrl": "ignored"},
                },
                {"name": "Bob Smith", "_id": "ignored"},
            ],
            "organization": {
                "name": "lab",
                "fullname": "Research Lab",
                "avatar": "ignored",
            },
        }
        for path, data in (
            ("/api/daily_papers", [{"paper": paper}]),
            ("/api/papers/search", [paper]),
            ("/api/papers/2602.08025", paper),
        ):
            output = compact_result(path, {"ok": True, "data": data})["data"]
            item = output[0] if isinstance(output, list) else output
            self.assertEqual(item["ai_summary"], "AI overview")
            self.assertEqual(item["ai_keywords"], ["agent", "memory"])
            self.assertEqual(item["authors"], ["Alice Smith", "Bob Smith"])
            self.assertNotIn("avatar", item["organization"])

    def test_discovery_signals_in_every_paper_tool(self):
        paper = {
            "id": "2602.08025",
            "upvotes": 130,
            "githubStars": 0,
            "githubRepo": "https://github.com/lab/project",
            "projectPage": "https://example.org",
            "submittedOnDailyBy": {"name": "alice"},
            "submittedOnDailyAt": "2026-02-09",
            "numTotalModels": 3,
            "numTotalDatasets": 0,
            "ai_summary_model": "summary-model",
        }
        for path, data in (
            (
                "/api/daily_papers",
                [{"paper": paper, "numComments": 2, "isAuthorParticipating": False}],
            ),
            ("/api/papers/search", [{"paper": paper, "numComments": 0}]),
            ("/api/papers/2602.08025", paper),
        ):
            result = compact_result(path, {"ok": True, "data": data})["data"]
            item = result[0] if isinstance(result, list) else result
            self.assertEqual(item["upvotes"], 130)
            self.assertEqual(item["githubStars"], 0)
            self.assertEqual(item["githubRepo"], paper["githubRepo"])
            self.assertEqual(item["projectPage"], paper["projectPage"])
            self.assertEqual(item["submitted_by"]["name"], "alice")
            self.assertEqual(item["numTotalDatasets"], 0)
            if path.endswith("search"):
                self.assertEqual(item["numComments"], 0)
            elif path.endswith("daily_papers"):
                self.assertFalse(item["isAuthorParticipating"])
            else:
                self.assertNotIn("numComments", item)

    def test_errors_and_markdown_unchanged(self):
        error = {"ok": False, "error": "404"}
        self.assertEqual(compact_result("/api/papers/search", error), error)
        text = {"ok": True, "data": "# Original paper"}
        self.assertEqual(compact_result("/papers/2602.08025.md", text), text)


if __name__ == "__main__":
    unittest.main()
