import os
from typing import TypeVar, get_args, get_origin

from pydantic import BaseModel

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.util.file_manipulation import add_to_jsonl_file
from forecasting_tools.util.misc import clean_indents

T = TypeVar("T")

DEFAULT_STRUCTURE_OUTPUT_MODEL = GeneralLlm(
    "openrouter/openai/gpt-4.1-mini", temperature=0.5
)


async def structure_output(
    text_to_structure: str,
    output_type: type[T],
    model: GeneralLlm | str = DEFAULT_STRUCTURE_OUTPUT_MODEL,
    num_validation_samples: int = 1,
    allowed_tries: int = 3,  # Allowed tries per sample
    additional_instructions: str | None = None,
) -> T:
    if num_validation_samples < 1:
        raise ValueError("Number of samples must be at least 1")
    samples = []
    for _ in range(num_validation_samples):
        sample = await _structure_output_single_sample(
            text_to_structure,
            output_type,
            model,
            allowed_tries,
            additional_instructions,
        )
        samples.append(sample)
    first_sample = samples[0]
    for i, sample in enumerate(samples):
        if sample != first_sample:
            raise ValueError(
                f"Sampled outputs are not the same:\nFirst Sample:\n{first_sample}\n\nSample {i + 1}:\n{sample}"
            )
    return first_sample


async def _structure_output_single_sample(
    text_to_structure: str,
    output_type: type[T],
    model: GeneralLlm | str = DEFAULT_STRUCTURE_OUTPUT_MODEL,
    allowed_tries: int = 3,
    additional_instructions: str | None = None,
) -> T:
    if not text_to_structure:
        raise ValueError("Output is empty")
    # Initialize with empty instructions
    pydantic_instructions = ""

    # Check if output_type is directly a BaseModel subclass
    try:
        if issubclass(output_type, BaseModel):
            pydantic_instructions = (
                GeneralLlm.get_schema_format_instructions_for_pydantic_type(output_type)
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
                    pydantic_instructions = (
                        GeneralLlm.get_schema_format_instructions_for_pydantic_type(
                            item_type
                        )
                    )
            except TypeError:
                pass

    type_not_found = "<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>"
    prompt = clean_indents(
        f"""
        You are a data analyst helping to convert text into structured data.
        You will receive text in between a bunch of <><><><><><><><><><><><> (each with 'start' and 'end' tags)
        Please convert the text to the following python parsable type:
        {output_type}

        General instructions:
        - When you give your answer, give no reasoning. Just output the final type w/o any other words.
        - If the type requires fields (e.g. dict or pydantic type):
            - Please return a JSON object (i.e. a dict)
            - Only fill in fields that are explicitly given in the text
            - Do not guess the fields
            - Do not fill in fields that are not explicitly given in the text
        - Do not summarize any of the text. Only give direct quotes (with only slight formatting changes)
        - If the text is completely unrelated to the requested type please just say "{type_not_found}"
        - If 'final answers' are mentioned, please prioritize using them to fill your structured response (i.e. avoid using intermediary steps)
        - Do not exclude any misc information that might be difficult to copy (assuming its part of the intended answer)
            - For example, include full URLs if they are provided with the intended answer. Copy all them exactly as they are shown in the text. Do not skip them.

        {"Additional instructions from the your manager:" if additional_instructions else ""}
        {additional_instructions if additional_instructions else ""}

        Here is the text:

        <><><><><><><><><><><> START TEXT <><><><><><><><><><><>



        {text_to_structure}



        <><><><><><><><><><><> END TEXT <><><><><><><><><><><>


        Please convert the text to the following type:
        {output_type}

        {pydantic_instructions}


        Please return an answer in the format given to you, and remember to keep URLs in text if they are included as part of the intended answer!
        """
    ).strip()
    # Edge cases that this prompt only mostly handles (in the end these will error out):
    # - When a binary forecast is asked for, but the response refuses to give one (e.g. because its too uncertain) then the forecast is set to "null" despite that not being an allowed value
    # - When a numeric forecast is requested, but the response only gives a single number (not a range of percentiles) then it will give a single percentile of {percentile: 0, value: number}

    llm = GeneralLlm.to_llm(model)

    result = await llm.invoke_and_return_verified_type(
        prompt,
        output_type,
        allowed_invoke_tries_for_failed_output=allowed_tries,
    )

    SAVE_STRUCTURE_OUTPUT = os.getenv("SAVE_STRUCTURE_OUTPUT")
    if SAVE_STRUCTURE_OUTPUT and SAVE_STRUCTURE_OUTPUT.lower() == "true":
        file_name = "structure_output_examples.jsonl"
        result_as_json = ""
        if isinstance(result, BaseModel):
            result_as_json = result.model_dump()
        elif isinstance(result, list):
            result_as_json = [item.model_dump() for item in result]
        else:
            result_as_json = result

        add_to_jsonl_file(
            file_name,
            [
                {
                    "text_to_structure": text_to_structure,
                    "output_type": output_type.__name__,
                    "additional_instructions": additional_instructions,
                    "result": result_as_json,
                }
            ],
        )

    return result
