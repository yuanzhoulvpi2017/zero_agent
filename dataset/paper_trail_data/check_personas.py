"""Persona sampling and dialogue-chain projection checks."""

import unittest
from pathlib import Path
import sys

# Allow `python dataset/paper_trail_data/check_personas.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_trail_data.chain import build_dialogue_chain
from paper_trail_data.personas import (
    MAX_CHAIN_TURNS,
    catalog,
    categories,
    move_instruction,
    sample_batch,
    sample_persona,
)


class PersonaChecks(unittest.TestCase):
    def test_catalog_and_sample_cover_roles(self):
        self.assertEqual(len(categories()), 8)
        self.assertGreaterEqual(len(catalog()), 16)
        batch = sample_batch(8, seed=1, max_turns=5)
        self.assertEqual(len(batch), 8)
        self.assertEqual(len({item["category"] for item in batch}), 8)
        for item in batch:
            self.assertLessEqual(item["max_turns"], MAX_CHAIN_TURNS)
            self.assertEqual(len(item["jump_plan"]), item["max_turns"])
            self.assertTrue(item["jump_plan"][0] == "open" or "open" in item["jump_plan"])

    def test_moves_include_person_questions(self):
        persona = sample_persona(__import__("random").Random(3), max_turns=5)
        text = move_instruction("ask_identity", persona)
        self.assertIn("叫什么", text)
        open_guide = move_instruction("open", persona)
        self.assertIn(persona["topic"], open_guide)
        self.assertIn("不要", open_guide)

    def test_persona_prompt_hides_brief(self):
        from paper_trail_data.personas import persona_system_prompt

        persona = sample_persona(__import__("random").Random(5), max_turns=3)
        prompt = persona_system_prompt(persona)
        # 新 prompt 版本强调短聊、别复述、别连环铺垫
        self.assertIn("别一次讲全", prompt)
        self.assertIn("不要复述出来", prompt)
        self.assertIn("禁止复述人设", prompt)
        self.assertIn("连环铺垫", prompt)
        self.assertIn("不要用 emoji", prompt)
        self.assertIn("先别/先别管", prompt)
        self.assertIn("markdown", prompt)
        self.assertIn(persona["goal"], prompt)

    def test_dialogue_chain_keeps_message_and_llm_sides(self):
        trajectory = [
            {"id": "u1", "role": "user", "content": "你是谁？", "timestamp": "t1"},
            {
                "id": "a1",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "介绍自己"},
                    {"type": "text", "text": "我是小埋。"},
                ],
                "timestamp": "t2",
            },
            {"id": "u2", "role": "user", "content": "再推荐一篇", "timestamp": "t3"},
            {
                "id": "a2",
                "role": "assistant",
                "content": [{"type": "text", "text": "可以看这篇。"}],
                "timestamp": "t4",
            },
        ]
        llm_calls = [
            {
                "call_index": 0,
                "turn_index": 0,
                "request": {"messages": [{"role": "user", "content": "你是谁？"}]},
                "response": {"text": "我是小埋。", "thinking": "介绍自己", "tool_calls": []},
            },
            {
                "call_index": 1,
                "turn_index": 1,
                "request": {"messages": [{"role": "user", "content": "再推荐一篇"}]},
                "response": {"text": "可以看这篇。", "thinking": "", "tool_calls": []},
            },
        ]
        chain = build_dialogue_chain(
            session_id="a" * 32,
            trajectory=trajectory,
            llm_calls=llm_calls,
            persona={"id": "grad-lost-1"},
            moves=["ask_identity", "narrow"],
            manifest={"status": "completed", "config": {"model": "deepseek-v4-flash"}},
        )
        self.assertEqual(chain["turn_count"], 2)
        self.assertEqual(chain["turns"][0]["move"], "ask_identity")
        self.assertTrue(chain["turns"][0]["assistant"]["thinking_present"])
        self.assertEqual(chain["turns"][0]["llm_calls"][0]["request"]["messages"][0]["content"], "你是谁？")
        self.assertEqual(chain["manifest"]["model"], "deepseek-v4-flash")
        self.assertLessEqual(chain["turn_count"], MAX_CHAIN_TURNS)


if __name__ == "__main__":
    unittest.main()
