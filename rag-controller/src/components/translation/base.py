from abc import abstractmethod, ABC
from src.settings.settings import TranslationTokenizerSettings


class BaseTranslation(ABC):
    def __init__(
        self,
        translation_tokenizer_settings: TranslationTokenizerSettings,
    ):
        self.translation_tokenizer_settings = translation_tokenizer_settings

    @abstractmethod
    def preprocess_message_for_translation(self, *args, **kwargs):
        pass

    @abstractmethod
    def postprocess_message_for_translation(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_input_ids(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_prefix_ids(self, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    @abstractmethod
    def translation_pipeline(self, *args, **kwargs):
        pass