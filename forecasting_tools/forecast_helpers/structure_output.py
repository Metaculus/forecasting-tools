from typing import TypeVar, get_args, get_origin

from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm

T = TypeVar("T")


async def structure_output(
    output: str,
    output_type: type[T],
    model: GeneralLlm | str = "openrouter/openai/gpt-4.1-mini",
    allowed_tries: int = 3,
) -> T:
    if not output:
        raise ValueError("Output is empty")
    # Initialize with empty instructions
    pydantic_instructions = ""

    # Check if output_type is directly a BaseModel subclass
    try:
        if issubclass(output_type, BaseModel):
            pydantic_instructions = (
                GeneralLlm.get_schema_format_instructions_for_pydantic_type(
                    output_type
                )
            )
    except TypeError:
        # Not a class, might be a generic type like list[BaseModel]
        pass

    # Check if output_type is list[BaseModel]
    origin = get_origin(output_type)
    if origin is list:
        args = get_args(output_type)
        if args and len(args) == 1:
            item_type = args[0]
            try:
                if issubclass(item_type, BaseModel):
                    pydantic_instructions = GeneralLlm.get_schema_format_instructions_for_pydantic_type(
                        item_type
                    )
            except TypeError:
                pass

    type_not_found = "<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>"
    prompt = clean_indents(
        f"""
        You are a secretary helping to convert text into structured data.
        You will receive text in between a bunch of <><><><><><><><><><><><> (each with 'start' and 'end' tags)
        Please convert the text to the following python parsable type:
        {output_type}

        When you give your answer, give no reasoning. Just output the final type w/o any other words.
        If the type requires fields (e.g. dict or pydantic type):
        - Please return a JSON object (i.e. a dict)
        - Only fill in fields that are explicitly given and required in the text
        - Do not guess the fields
        - Do not fill in fields that are not explicitly given and required in the text
        - Do not summarize any of the text. Only give direct quotes (with only slight formatting changes)
        - If the text is completely unrelated to the requested type please just say "{type_not_found}"
        - DO NOT exclude links if links are provided with the intended answer please keep the links. Copy all them as they are shown in the text.

        If 'final answers' are mentioned, please prioritize using them to fill your structured response (i.e. avoid using intermediary steps)
        Here is the text:

        <><><><><><><><><><><> START TEXT <><><><><><><><><><><>



        {output}



        <><><><><><><><><><><> END TEXT <><><><><><><><><><><>


        Please convert the text to the following type:
        {output_type}

        {pydantic_instructions}


        Please return an answer in the format given to you, and remember to include links if they are included!
        """
    ).strip()

    llm = GeneralLlm.to_llm(model)

    result = await llm.invoke_and_return_verified_type(
        prompt,
        output_type,
        allowed_invoke_tries_for_failed_output=allowed_tries,
    )

    return result
