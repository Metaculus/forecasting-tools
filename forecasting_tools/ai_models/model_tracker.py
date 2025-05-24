import logging

from litellm import model_cost

logger = logging.getLogger(__name__)


class TrackedModel:
    def __init__(self, model: str) -> None:
        self.model = model
        self.gave_cost_tracking_warning = False


class ModelTracker:
    _model_trackers: dict[str, TrackedModel] = {}

    @classmethod
    def give_cost_tracking_warning_if_needed(cls, model: str) -> None:
        model_tracker = cls._model_trackers.get(model)
        if model_tracker is None:
            cls._model_trackers[model] = TrackedModel(model)
        model_tracker = cls._model_trackers[model]

        if model_tracker.gave_cost_tracking_warning:
            return

        assert isinstance(model_cost, dict)
        supported_model_names = model_cost.keys()
        model_not_supported = model not in supported_model_names
        if model_not_supported:
            message = f"Warning: Model {model} does not support cost tracking."
            logger.warning(message)

        model_tracker.gave_cost_tracking_warning = True
