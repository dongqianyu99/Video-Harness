import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer


class _FakeSentencePiece:
    def __init__(self, encoded_tokens: list[int]):
        self.encoded_tokens = encoded_tokens
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        self.calls.append((text, add_bos))
        return list(self.encoded_tokens)


def _make_uninitialized_tokenizer(fake_sentencepiece: _FakeSentencePiece, max_len: int):
    tokenizer = object.__new__(_tokenizer.PaligemmaTokenizer)
    tokenizer._max_len = max_len  # noqa: SLF001
    tokenizer._tokenizer = fake_sentencepiece  # noqa: SLF001
    tokenizer._cache_digest = "test-digest"  # noqa: SLF001
    return tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_tokenize_text_is_raw_bos_padded_and_typed():
    text = "Observed contact with the handle."
    fake_sentencepiece = _FakeSentencePiece([2, 17, 23])
    tokenizer = _make_uninitialized_tokenizer(fake_sentencepiece, max_len=6)

    tokens, mask = tokenizer.tokenize_text(text)

    assert fake_sentencepiece.calls == [(text, True)]
    assert tokens.tolist() == [2, 17, 23, 0, 0, 0]
    assert mask.tolist() == [True, True, True, False, False, False]
    assert tokens.dtype == np.int32
    assert mask.dtype == np.bool_
    assert tokenizer.cache_digest == "test-digest"


def test_tokenize_text_rejects_overflow_instead_of_truncating():
    fake_sentencepiece = _FakeSentencePiece([2, 3, 4, 5])
    tokenizer = _make_uninitialized_tokenizer(fake_sentencepiece, max_len=3)

    with pytest.raises(ValueError, match=r"max_len|overflow|long"):
        tokenizer.tokenize_text("too many tokens")


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
