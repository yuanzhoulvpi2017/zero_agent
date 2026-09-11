"""Deterministic HF output projection; the disk cache retains the raw response."""

from copy import deepcopy


def clip(value, limit):
    text = value if isinstance(value, str) else ""
    return text[:limit] + ("…" if len(text) > limit else "")


def identity(value):
    if not value:
        return None
    if isinstance(value, str):
        return {"name": clip(value, 160)}
    if not isinstance(value, dict):
        return None
    name = clip(value.get("name"), 160)
    return {
        "name": name,
        "display_name": clip(value.get("fullname") or name, 200),
        "url": "https://huggingface.co/" + name if name else None,
    }


def paper_summary(item, detailed=False):
    paper = item.get("paper", item)
    if not isinstance(paper, dict):
        paper = item
    pid = clip(paper.get("id", item.get("id", "")), 64)
    abstract = paper.get("summary") or item.get("summary") or ""
    ai_summary = paper.get("ai_summary") or item.get("ai_summary") or ""
    keywords = paper.get("ai_keywords") or item.get("ai_keywords") or []
    authors = paper.get("authors") or item.get("authors") or []
    length = 4000
    result = {
        "id": pid,
        "title": clip(paper.get("title") or item.get("title"), 300),
        "published_at": clip(paper.get("publishedAt"), 40),
        "url": "https://huggingface.co/papers/" + pid,
        "abstract": clip(abstract, length),
        "abstract_truncated": len(abstract) > length,
        "ai_summary": clip(ai_summary, length),
        "ai_summary_truncated": len(ai_summary) > length,
        "ai_keywords": [keyword for keyword in keywords if isinstance(keyword, str)]
        if isinstance(keywords, list)
        else keywords,
        "authors": [
            clip(a.get("name") if isinstance(a, dict) else a, 200)
            for a in authors[:12]
            if isinstance(a, (str, dict))
        ],
        "authors_omitted": max(0, len(authors) - 12),
        "submitted_by": identity(
            item.get("submittedBy")
            or paper.get("submittedBy")
            or paper.get("submittedOnDailyBy")
        ),
        "organization": identity(item.get("organization") or paper.get("organization")),
    }
    if item.get("paper"):
        result["feed_published_at"] = clip(item.get("publishedAt"), 40)
    # Keep discovery and reproducibility signals in both lists and details.
    # Missing is different from zero: never invent a zero count.
    for key in (
        "upvotes",
        "numComments",
        "githubStars",
        "isAuthorParticipating",
        "submittedOnDailyAt",
        "githubRepo",
        "projectPage",
        "ai_summary_model",
        "numTotalModels",
        "numTotalDatasets",
        "numTotalSpaces",
    ):
        value = item.get(key)
        if value is None:
            value = paper.get(key)
        if isinstance(value, (str, int, float, bool)):
            result[key] = clip(value, 500) if isinstance(value, str) else value
    return result


def compact_result(path, result):
    if not result.get("ok") or "data" not in result:
        return result
    # Never mutate cached raw data, including nested objects.
    output = {k: deepcopy(v) for k, v in result.items() if k != "data"}
    data = result["data"]
    if path in {"/api/daily_papers", "/api/papers/search"} and isinstance(data, list):
        output["data"] = [
            paper_summary(item) for item in data[:20] if isinstance(item, dict)
        ]
        output["returned"] = len(output["data"])
        output["omitted"] = max(0, len(data) - 20)
        output["detail_hint"] = (
            "精简摘要可能截断；用 paper_metadata 查询选中论文，read_paper 分段读原文。"
        )
    elif path.startswith("/api/papers/") and isinstance(data, dict):
        output["data"] = paper_summary(data, detailed=True)
        output["detail_hint"] = "仅保留核心元数据；摘要截断时用 read_paper 继续读取。"
    elif path in {"/api/models", "/api/datasets", "/api/spaces"} and isinstance(
        data, list
    ):
        kind = path.rsplit("/", 1)[-1]
        output["data"] = []
        for item in data[:20]:
            if not isinstance(item, dict):
                continue
            identifier = clip(item.get("id"), 200)
            resource = {
                "id": identifier,
                "url": "https://huggingface.co/"
                + ("" if kind == "models" else kind + "/")
                + identifier,
            }
            for key in ("pipeline_tag", "downloads", "likes"):
                value = item.get(key)
                if isinstance(value, (str, int, float)):
                    resource[key] = (
                        clip(value, 100) if isinstance(value, str) else value
                    )
            output["data"].append(resource)
        output["omitted"] = max(0, len(data) - 20)
    else:
        return result
    output["compacted"] = True
    return output
