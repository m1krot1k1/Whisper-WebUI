try:
    from gradio_i18n import Translate, gettext as _
    AUTOMATIC_DETECTION = _("Automatic Detection")
except (ImportError, LookupError):
    # Fallback when gradio_i18n is not available or context is missing
    def _(text):
        return text
    AUTOMATIC_DETECTION = "Automatic Detection"
GRADIO_NONE_STR = ""
GRADIO_NONE_NUMBER_MAX = 9999
GRADIO_NONE_NUMBER_MIN = 0
