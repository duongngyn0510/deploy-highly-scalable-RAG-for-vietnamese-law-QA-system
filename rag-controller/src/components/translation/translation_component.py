import logging

from injector import inject, singleton
from src.settings.settings import Settings
from src.components.translation.base import BaseTranslation

logger = logging.getLogger(__name__)


@singleton
class TranslationComponent:
    translation: BaseTranslation

    @inject
    def __init__(self, settings: Settings) -> None:
        if settings.translation.enabled == True:
            translation_mode = settings.translation.mode
            logger.info("Initializing the translation in mode=%s", translation_mode)

            match translation_mode:
                case "triton_nllb":
                    from src.components.translation.triton.nllb import (
                        NllbTritonServerTranslation,
                    )

                    translation_tokenizer_settings = settings.translation_tokenizer
                    nllb_translation_triton_server_settings = settings.nllb_translation_triton_server
                    self.translation = NllbTritonServerTranslation(
                        translation_tokenizer_settings,
                        nllb_translation_triton_server_settings,
                    )

                case "triton_envit5":
                    from src.components.translation.triton.envit5 import (
                        Envit5TritonServerTranslation
                    )

                    translation_tokenizer_settings = settings.translation_tokenizer
                    envit5_translation_triton_server_settings = settings.envit5_translation_triton_server
                    self.translation = Envit5TritonServerTranslation(
                        translation_tokenizer_settings,
                        envit5_translation_triton_server_settings,
                    )