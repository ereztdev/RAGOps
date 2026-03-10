"""Tests for core.tokenizer (shared whitespace tokenizer for chunking and BM25)."""
from __future__ import annotations

import unittest

from core.tokenizer import simple_tokenize


class TestSimpleTokenize(unittest.TestCase):
    def test_empty_string_returns_empty_list(self) -> None:
        self.assertEqual(simple_tokenize(""), [])

    def test_whitespace_only_returns_empty_list(self) -> None:
        self.assertEqual(simple_tokenize("   \n\t  "), [])

    def test_single_word(self) -> None:
        self.assertEqual(simple_tokenize("hello"), ["hello"])

    def test_multiple_words(self) -> None:
        self.assertEqual(simple_tokenize("hello world"), ["hello", "world"])

    def test_deterministic_same_input_same_output(self) -> None:
        text = "The overvoltage protection is set to trigger at 130V"
        a = simple_tokenize(text)
        b = simple_tokenize(text)
        self.assertEqual(a, b)
