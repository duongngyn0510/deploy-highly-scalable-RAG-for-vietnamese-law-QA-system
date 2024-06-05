from abc import abstractmethod, ABC
from src.settings.settings import TranslationSettings, TranslationTokenizerSettings


class BaseTranslation(ABC):
    def __init__(
        self,
        translate_settings: TranslationSettings,
        translate_tokenizer_settings: TranslationTokenizerSettings,
    ):
        self.translate_settings = translate_settings
        self.translate_tokenizer_settings = translate_tokenizer_settings

    @abstractmethod
    def get_tokenizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_input_ids(
        self, tokenizer: get_tokenizer, sentences: list[str], *args, **kwargs
    ):
        pass

    @abstractmethod
    def get_prefix_ids(
        self,
        tokenizer: get_tokenizer,
        sentences: list[str],
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    def infer(self, input_ids, prefix_ids, *args, **kwargs):
        pass

    @abstractmethod
    def decode(
        self, 
        tokenizer: get_tokenizer, 
        outputs: infer, 
        *args, 
        **kwargs
    ):
        pass

