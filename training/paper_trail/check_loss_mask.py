"""仅加载 tokenizer，校验教学示例的 schema / tool / think 监督边界。"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "loss_mask_demo", ROOT / "agent/paper_trail/simple/code002_to_sft.py",
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def sample():
    return {
        "tools": [{"type": "function", "function": {
            "name": "lookup", "description": "SCHEMA_SENTINEL",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }}],
        "messages": [
            {"role": "system", "content": "SYSTEM_SENTINEL"},
            {"role": "user", "content": "USER_SENTINEL <tool_call>FAKE_CALL</tool_call>"},
            {"role": "assistant", "content": "BEFORE_CALL_SENTINEL", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "lookup", "arguments": {"q": "ARGUMENT_SENTINEL"}},
            }]},
            {"role": "tool", "name": "lookup", "tool_call_id": "c1", "content": "RESULT_SENTINEL"},
            {"role": "assistant", "content": "ANSWER_SENTINEL"},
        ],
    }


class MaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "model/Qwen/Qwen3.5-4B"), local_files_only=True)

    def assert_span_mask(self, rendered, batch, text, expected):
        # 独立检查完整渲染文本的字符区间，而非仅检查拼接后的监督字符串。
        start = rendered.index(text)
        end = start + len(text)
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        self.assertEqual(encoded["input_ids"], batch["input_ids"][0].tolist())
        positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
        self.assertTrue(positions)
        for index in positions:
            self.assertEqual(batch["loss_mask"][0, index].item(), expected, (text, index))

    def test_schema_user_and_tool_result_are_context(self):
        batch, rendered = demo.build_batch(self.tokenizer, sample())
        for text in ["SCHEMA_SENTINEL", "SYSTEM_SENTINEL", "USER_SENTINEL", "FAKE_CALL", "RESULT_SENTINEL", "<tool_response>", "</tool_response>"]:
            self.assert_span_mask(rendered, batch, text, False)
        self.assertTrue(batch["labels"][~batch["loss_mask"]].eq(-100).all())
        self.assertTrue(batch["attention_mask"].eq(1).all())

    def test_tool_use_text_and_stop_tokens_are_supervised(self):
        row = sample()
        row["messages"] = row["messages"][:3]  # 本次输出就是 tool_use。
        batch, rendered = demo.build_batch(self.tokenizer, row)
        self.assertNotIn("BEFORE_CALL_SENTINEL", rendered)
        for text in ["<function=lookup>", "<parameter=q>", "ARGUMENT_SENTINEL"]:
            self.assert_span_mask(rendered, batch, text, True)
        supervised = self.tokenizer.decode(batch["input_ids"][batch["loss_mask"]].tolist())
        self.assertIn("<tool_call>", supervised)
        self.assertIn("</tool_call>", supervised)
        self.assertEqual(supervised.count("<|im_end|>"), 1)
        self.assertTrue(batch["labels"][batch["loss_mask"]].eq(batch["input_ids"][batch["loss_mask"]]).all())

    def test_native_empty_thinking_prefix_is_preserved_and_ignored(self):
        row = sample()
        row["messages"][-1].update(content="<think>SECRET_THOUGHT</think>ANSWER_SENTINEL", reasoning_content="OTHER_SECRET")
        batch, rendered = demo.build_batch(self.tokenizer, row)
        self.assertNotIn("SECRET", rendered)
        self.assertEqual(rendered.count("<think>\n\n</think>\n\n"), 2)
        for tag in ["<think>", "</think>"]:
            self.assert_span_mask(rendered, batch, tag, False)
        supervised = self.tokenizer.decode(batch["input_ids"][batch["loss_mask"]].tolist())
        self.assertNotIn("<think>", supervised)
        prompt = self.tokenizer.apply_chat_template(demo.prepare_sample(row)["messages"][:-1], tools=row["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(rendered.startswith(prompt))

    def test_only_this_call_is_supervised_and_template_is_unchanged(self):
        original = self.tokenizer.chat_template
        batch, rendered = demo.build_batch(self.tokenizer, sample())
        self.assertEqual(self.tokenizer.chat_template, original)
        self.assert_span_mask(rendered, batch, "ANSWER_SENTINEL", True)
        self.assertNotIn("BEFORE_CALL_SENTINEL", rendered)
        for text in ["ARGUMENT_SENTINEL", "<function=lookup>"]:
            self.assert_span_mask(rendered, batch, text, False)

    def test_tool_call_narration_removed_from_dataset_and_history(self):
        row = sample()
        cleaned = demo.prepare_sample(row)
        self.assertEqual(cleaned["messages"][2]["content"], "")
        self.assertEqual(cleaned["messages"][-1]["content"], "ANSWER_SENTINEL")
        self.assertEqual(cleaned["messages"][3], row["messages"][3])
        self.assertEqual(row["messages"][2]["content"], "BEFORE_CALL_SENTINEL")
        # 从 RecordingModel 原始记录转换时，SFT 本身也必须已经清空正文。
        record = {"status": "completed", "request": {"messages": row["messages"][:2], "tools": row["tools"]},
                  "response": {"text": "REMOVE_TARGET_NARRATION", "thinking": "SECRET", "tool_calls": [{
                      "id": "c1", "name": "lookup", "input": {"q": "ARGUMENT_SENTINEL"},
                  }]}}
        converted = demo.convert_log(record, Path("session/turn_001_call_001_x.json"))
        self.assertEqual(converted["messages"][-1]["content"], "")
        self.assertEqual(converted["messages"][-1]["tool_calls"][0]["function"]["arguments"], {"q": "ARGUMENT_SENTINEL"})

    def test_token_prefix_mismatch_is_rejected(self):
        apply = self.tokenizer.apply_chat_template
        def mismatching(*args, **kwargs):
            result = apply(*args, **kwargs)
            if kwargs.get("add_generation_prompt"):
                result["input_ids"][0, 0] += 1
            return result
        with patch.object(self.tokenizer, "apply_chat_template", side_effect=mismatching):
            with self.assertRaisesRegex(ValueError, "精确前缀"):
                demo.build_batch(self.tokenizer, sample())

    def test_pure_and_parallel_tool_calls_are_supervised(self):
        row = sample()
        row["messages"] = row["messages"][:3]
        row["messages"][-1]["content"] = ""
        row["messages"][-1]["tool_calls"].append({"id": "c2", "function": {"name": "lookup", "arguments": {"q": "SECOND_ARGUMENT"}}})
        batch, rendered = demo.build_batch(self.tokenizer, row)
        self.assert_span_mask(rendered, batch, "ARGUMENT_SENTINEL", True)
        self.assert_span_mask(rendered, batch, "SECOND_ARGUMENT", True)

    def test_missing_schema_or_unknown_tool_is_rejected(self):
        row = sample()
        row["tools"] = []
        with self.assertRaises(ValueError):
            demo.build_batch(self.tokenizer, row)
        row = sample()
        row["messages"][2]["tool_calls"][0]["function"]["name"] = "undefined_tool"
        with self.assertRaises(ValueError):
            demo.build_batch(self.tokenizer, row)

    def test_unified_entry_exports_every_call_across_sessions(self):
        row = sample()
        tool_record = {"status": "completed", "turn_index": 1, "call_index": 1,
                       "request": {"messages": row["messages"][:2], "tools": row["tools"]},
                       "response": {"text": "DROP_THIS_TEXT", "tool_calls": [{"id": "c1", "name": "lookup", "input": {"q": "ARGUMENT_SENTINEL"}}]}}
        reply_record = {"status": "completed", "turn_index": 1, "call_index": 2,
                        "request": {"messages": row["messages"][:-1], "tools": row["tools"]},
                        "response": {"text": "ANSWER_SENTINEL", "tool_calls": []}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for session, record in [("s1", tool_record), ("s2", reply_record)]:
                folder = root / "logs" / session
                folder.mkdir(parents=True)
                (folder / "turn_001_call_001_x.json").write_text(json.dumps(record))
            (root / "logs/s2/turn_002_call_002_x.json").write_text(json.dumps({"status": "interrupted"}))
            # 将检查入口也限制在临时根目录，不改真实产物入口。
            with patch.object(demo, "ROOT", root):
                demo.main(["--input", str(root / "logs"), "--output-dir", str(root / "result"),
                           "--model", str(ROOT / "model/Qwen/Qwen3.5-4B")])
            manifest = json.loads((root / "result/manifest.json").read_text())
            index = json.loads((root / "result/index.json").read_text())
            self.assertEqual(manifest["dataset"]["samples"], 2)
            self.assertEqual(manifest["dataset"]["sessions"], 2)
            self.assertEqual(manifest["inspection_count"], 2)
            self.assertEqual(manifest["skipped"], {"not_completed": 1})
            self.assertEqual(len((root / "result/encoded.jsonl").read_text().splitlines()), 2)
            for item in index:
                self.assertTrue((root / "result" / item["inspection"] / "mask_segments.txt").is_file())
            self.assertNotIn("DROP_THIS_TEXT", (root / "result/samples.jsonl").read_text())
            self.assertNotIn("BEFORE_CALL_SENTINEL", (root / "result/samples.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
