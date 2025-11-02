# Expose all interfaces for clean import
from .i_title_extractor import ITitleExtractor
from .i_author_extractor import IAuthorExtractor
from .i_year_extractor import IYearExtractor
#from .i_abstract_extractor import IAbstractExtractor
from .i_language_detector import ILanguageDetector
from .i_file_size_extractor import IFileSizeExtractor
#from .i_has_ocr_extractor import IHasOcrExtractor

__all__ = [
    "ITitleExtractor",
    "IAuthorExtractor",
    "IYearExtractor",
    #"IAbstractExtractor",
    "ILanguageDetector",
    "IFileSizeExtractor",
    #"IHasOcrExtractor",
]
