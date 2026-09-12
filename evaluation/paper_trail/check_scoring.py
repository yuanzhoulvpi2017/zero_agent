"""Checks for judge transcript tool evidence and automatic grounding."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scoring import (
    DIMENSIONS,
    automatic_metrics,
    compact_session,
    papers_from_payload,
)


def _session_dir(root: Path) -> Path:
    directory = root / "session"
    directory.mkdir()
    (directory / "persona.json").write_text(
        json.dumps(
            {
                "id": "grad-demo-1",
                "name": "小林",
                "category": "graduate_student",
                "category_label": "在读硕士",
                "knowledge": "medium",
                "topic": "Agent 记忆",
                "voice": "短",
                "goal": "写开题",
                "jump_plan": ["ask", "narrow"],
                "max_turns": 2,
            },
            ensure_ascii=False,
        )
    )
    (directory / "manifest.json").write_text(
        json.dumps({"status": "completed", "session_usage": {}}, ensure_ascii=False)
    )
    payload = {
        "ok": True,
        "data": [
            {
                "id": "2609.11561",
                "title": "Memory as Plans: World-Action Modeling",
            }
        ],
    }
    (directory / "dialogue_chain.json").write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "turn_index": 0,
                        "move": "ask",
                        "user": {"text": "记忆方向有啥论文"},
                        "assistant": {"text": "检索到 2609.11561，另外还有 2602.00001。"},
                        "agent_messages": [
                            {
                                "content": [
                                    {"type": "tool_use", "id": "c1", "name": "search_papers"},
                                    {
                                        "type": "tool_result",
                                        "id": "c1",
                                        "name": "search_papers",
                                        "output": [
                                            {
                                                "type": "text",
                                                "text": json.dumps(payload, ensure_ascii=False),
                                            }
                                        ],
                                    },
                                ]
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )
    )
    return directory


class ScoringChecks(unittest.TestCase):
    def test_score_dimensions_are_the_three_product_axes(self):
        self.assertEqual(list(DIMENSIONS), ["grounding", "helpfulness", "dialogue"])
        self.assertTrue(DIMENSIONS["grounding"].startswith("依据"))
        self.assertTrue(DIMENSIONS["helpfulness"].startswith("有用"))
        self.assertTrue(DIMENSIONS["dialogue"].startswith("对话"))

    def test_papers_from_search_payload(self):
        papers = papers_from_payload(
            {
                "ok": True,
                "data": [{"id": "2512.13564", "title": "MaP-WAM"}],
                "arxiv_url": "https://arxiv.org/abs/2609.11561",
            },
            "",
        )
        self.assertEqual([item["id"] for item in papers], ["2512.13564", "2609.11561"])
        self.assertEqual(papers[0]["title"], "MaP-WAM")

    def test_compact_session_includes_tool_papers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _session_dir(Path(tmp))
            text = compact_session(directory, max_turns=2)
            self.assertIn("2609.11561", text)
            self.assertIn("Memory as Plans", text)
            self.assertIn("工具证据", text)
            self.assertIn("本场工具返回过的论文", text)
            self.assertIn("已经说出口的事", text)
            self.assertIn("在读硕士", text)

    def test_automatic_grounding_splits_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _session_dir(Path(tmp))
            metrics = automatic_metrics(directory, max_turns=2)
            self.assertEqual(metrics["reply_arxiv_ids"], ["2609.11561", "2602.00001"])
            self.assertEqual(metrics["arxiv_grounded"], 1)
            self.assertEqual(metrics["arxiv_ungrounded"], 1)
            self.assertEqual(metrics["ungrounded_arxiv_ids"], ["2602.00001"])


if __name__ == "__main__":
    unittest.main()
