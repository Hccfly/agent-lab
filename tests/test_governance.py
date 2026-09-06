"""路径、标识、原子写、并发事务、超时与限流测试。"""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}):
    import agent
    import governance
    import memory
    from governance import ExecutionGovernor, ExecutionTimeout, RateLimitExceeded
    from rag import VectorStore
    from tools import read_file


class StubStore:
    def __init__(self, _path):
        self.items = []

    def search(self, _query, top_k=3):
        return []

    def add(self, text, vector):
        self.items.append((text, vector))

    def save(self):
        pass

    def __len__(self):
        return len(self.items)


class GovernanceTests(unittest.TestCase):
    def test_read_file_allows_root_and_blocks_traversal_and_prefix_attack(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            allowed = base / "allowed"
            sibling = base / "allowed-evil"
            allowed.mkdir()
            sibling.mkdir()
            safe_file = allowed / "safe.txt"
            outside_file = sibling / "secret.txt"
            safe_file.write_text("safe", encoding="utf-8")
            outside_file.write_text("secret", encoding="utf-8")

            with patch.dict(os.environ, {"AGENT_ALLOWED_READ_ROOTS": str(allowed)}):
                self.assertEqual(read_file(str(safe_file)), "safe")
                with self.assertRaises(governance.PathAccessDenied):
                    read_file(str(allowed / ".." / "allowed-evil" / "secret.txt"))
                with self.assertRaises(governance.PathAccessDenied):
                    read_file(str(outside_file))

    def test_session_id_validation_prevents_file_escape(self):
        self.assertEqual(memory.validate_session_id("abc-1.test"), "abc-1.test")
        invalid = ["", ".hidden", "../escape", "..\\escape", "a/b", "has space", "a" * 129]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                memory.validate_session_id(value)

    def test_atomic_session_replace_failure_preserves_previous_json(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            memory, "SESSION_DIR", Path(directory)
        ):
            memory.save_session("atomic-1", "old", [{"role": "user", "content": "old"}])
            with patch.object(governance.os, "replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    memory.save_session(
                        "atomic-1", "new", [{"role": "user", "content": "new"}]
                    )
            restored = memory.load_session("atomic-1")
            self.assertEqual(restored["summary"], "old")
            self.assertEqual(restored["messages"][0]["content"], "old")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_same_session_concurrent_rounds_do_not_lose_messages(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            memory, "SESSION_DIR", Path(directory)
        ), patch.object(agent, "VectorStore", StubStore), patch.object(
            agent, "embed_texts", side_effect=lambda texts: [[1.0] for _ in texts]
        ):
            entered = threading.Barrier(2)

            def fake_create(**kwargs):
                user_messages = [
                    item["content"]
                    for item in kwargs["messages"]
                    if item["role"] == "user"
                ]
                if len(user_messages) == 1:
                    try:
                        entered.wait(timeout=0.2)
                    except threading.BrokenBarrierError:
                        pass
                message = SimpleNamespace(
                    content=f"answer:{user_messages[-1]}",
                    tool_calls=[],
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)],
                    usage=None,
                )

            answers = []

            def worker(question):
                answers.append(agent.run_agent(question, session_id="shared-1"))

            with patch.object(agent.client.chat.completions, "create", fake_create):
                threads = [
                    threading.Thread(target=worker, args=("first",)),
                    threading.Thread(target=worker, args=("second",)),
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=2)

            self.assertCountEqual(answers, ["answer:first", "answer:second"])
            saved = memory.load_session("shared-1")
            self.assertEqual(len(saved["messages"]), 4)
            self.assertCountEqual(
                [m["content"] for m in saved["messages"] if m["role"] == "user"],
                ["first", "second"],
            )

    def test_concurrent_vector_stores_merge_pending_items(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.json"
            count = 8
            ready = threading.Barrier(count)

            def worker(index):
                store = VectorStore(str(path))
                store.add(f"item-{index}", [float(index), 1.0])
                ready.wait(timeout=2)
                store.save()

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)

            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(stored), count)
            self.assertEqual(
                {item["text"] for item in stored},
                {f"item-{i}" for i in range(count)},
            )

    def test_execution_governor_times_out_and_keeps_slot_until_worker_finishes(self):
        governor = ExecutionGovernor(
            max_concurrency=1,
            timeout_seconds=0.05,
            queue_timeout_seconds=0.02,
        )
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        errors = []

        def blocking_operation():
            started.set()
            release.wait(timeout=2)
            finished.set()
            return "done"

        def first_caller():
            try:
                governor.run("slow-tool", blocking_operation)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=first_caller)
        thread.start()
        self.assertTrue(started.wait(timeout=1))
        with self.assertRaises(RateLimitExceeded):
            governor.run("second-tool", lambda: "should-not-run")
        thread.join(timeout=1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ExecutionTimeout)

        with self.assertRaises(RateLimitExceeded):
            governor.run("third-tool", lambda: "still-running")

        release.set()
        self.assertTrue(finished.wait(timeout=1))
        deadline = time.monotonic() + 1
        while True:
            try:
                result = governor.run("recovered-tool", lambda: "ok")
                break
            except RateLimitExceeded:
                if time.monotonic() >= deadline:
                    self.fail("后台工作结束后并发槽位没有释放")
        self.assertEqual(result, "ok")
        governor.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
