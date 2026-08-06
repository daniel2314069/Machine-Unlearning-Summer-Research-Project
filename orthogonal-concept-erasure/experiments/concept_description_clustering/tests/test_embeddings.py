from collections import Counter

from concept_clustering.embeddings import _selected_token_audit, shuffle_description_words
from concept_clustering.text_validation import words


class FakeTokenizer:
    model_max_length = 77

    def __call__(self, prompt, add_special_tokens=True, truncation=False):
        assert prompt == "Brown paws move quickly."
        return {"input_ids": [0, 10, 11, 12, 13, 1]}

    def convert_ids_to_tokens(self, token_ids):
        assert token_ids == [0, 10, 11, 12, 13, 1]
        return ["<|startoftext|>", "brown</w>", "paws</w>", "quickly</w>", ".</w>", "<|endoftext|>"]


def test_natural_audit_skips_terminal_punctuation():
    row = _selected_token_audit(
        FakeTokenizer(),
        "Brown paws move quickly.",
        "Brown paws move quickly.",
        "natural_last_token",
        "description",
        select_last_content_token=True,
    )
    assert row["selected_token"] == "quickly</w>"
    assert row["selected_token_position"] == 3


def test_word_shuffle_is_deterministic_and_preserves_bag():
    text = "A quiet whiskered creature balances carefully beside the sunny window."
    first = shuffle_description_words(text, "cat_001", 1729)
    second = shuffle_description_words(text, "cat_001", 1729)
    assert first == second
    assert first != text
    assert Counter(words(first)) == Counter(words(text))
