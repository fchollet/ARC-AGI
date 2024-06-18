import tokenizers
from dataclasses import dataclass


@dataclass
class TokenizerConfig:

    model_max_length: int = 1024
    padding: str = "max_length"
    truncation: str = "longest_first"
    max_length: int = 1024


@dataclass
class TokenizerOutput:

    input_ids: list
    attention_mask: list
    token_type_ids: list
    labels: list


class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizer = tokenizers.BertWordPieceTokenizer(
            "tokenizer/vocab.json", lowercase=True
        )

    @classmethod
    def from_pretrained(cls, model_name: str):
        config = TokenizerConfig()
        return cls(config)

    def __call__(self, text: str):

        encoded = self.tokenizer.encode(text)

        return TokenizerOutput(
            input_ids=encoded.ids,
            attention_mask=encoded.attention_mask,
            token_type_ids=encoded.type_ids,
            labels=encoded.ids,
        )
