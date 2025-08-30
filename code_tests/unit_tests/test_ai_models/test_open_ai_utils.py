from code_tests.unit_tests.test_ai_models.models_to_test import LLMTestData
from forecasting_tools.ai_models.ai_utils.openai_utils import OpenAiUtils


################################## Message Creation Tests ##################################
def test_user_message_creator_has_only_one_message() -> None:
    messages = OpenAiUtils.put_single_user_message_in_list_using_prompt("Hello")
    length_of_messages = len(messages)
    assert length_of_messages == 1, "Length of user message from prompt is not 1"


def test_system_and_user_message_creator_has_two_messages() -> None:
    messages = OpenAiUtils.create_system_and_user_message_from_prompt("Hello", "Hi")
    length_of_messages = len(messages)
    assert (
        length_of_messages == 2
    ), "Length of system and user message from prompt is not 2"


def test_vision_message_creator_has_one_message() -> None:
    vision_data = LLMTestData.CHEAP_VISION_MESSAGE_DATA
    messages = OpenAiUtils.put_single_image_message_in_list_using_gpt_vision_input(
        vision_data
    )
    length_of_messages = len(messages)
    assert length_of_messages == 1, "Length of vision message from prompt is not 1"


def test_system_and_vision_message_creator_has_two_messages() -> None:
    vision_data = LLMTestData.CHEAP_VISION_MESSAGE_DATA
    messages = OpenAiUtils.create_system_and_image_message_from_prompt(
        vision_data, "Hi"
    )
    length_of_messages = len(messages)
    assert (
        length_of_messages == 2
    ), "Length of system and vision message from prompt is not 2"
