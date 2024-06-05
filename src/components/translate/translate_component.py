import logging

from injector import inject, singleton
from src.settings.settings import Settings
from src.components.translate.base import BaseTranslation

logger = logging.getLogger(__name__)


@singleton
class TranslateComponent:
    translation: BaseTranslation

    @inject
    def __init__(self, settings: Settings) -> None:
        translate_mode = settings.translate.mode
        logger.info("Initializing the Translate in mode=%s", translate_mode)

        match translate_mode:
            case "triton":
                from src.components.translate.triton.base import TritonServerTranslation

                translation_settings = settings.translation
                translation_tokenizer_settings = settings.translation_tokenizer
                self.translation = TritonServerTranslation(translation_settings, translation_tokenizer_settings)

                    

                
