def check_valid_tokenizer(tokenizer) -> bool:
    """Check if the special tokens were correctly added to the tokenizer,
    and if they are not normalized.
    """
    tok_class = type(tokenizer).__name__.lower()
    if ("idefics" in tok_class) or ("mistral" in tok_class):
        assert "<image>" in tokenizer.get_vocab()
        assert "<fake_token_around_image>" in tokenizer.get_vocab()
        assert "<s>" in tokenizer.get_vocab()
        assert "</s>" in tokenizer.get_vocab()
        assert "<unk>" in tokenizer.get_vocab()

        for _, val in tokenizer.added_tokens_decoder.items():
            assert not val.normalized  # assert that normalized=False for all AddedToken
