import asyncio
import logging
import os
import random

import numpy as np
import requests
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from forecasting_tools.agents_and_tools.deprecated.configured_llms import BasicLlm
from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher
from forecasting_tools.util.misc import (
    clean_indents,
    raise_for_status_with_additional_info,
)

logger = logging.getLogger(__name__)


class Deduplicator:

    @classmethod
    async def deduplicate_list_in_batches(
        cls,
        items_to_deduplicate: list[str],
        prompt_context: str,
        initial_semantic_threshold: float = 0.85,
    ) -> list[str]:
        semantic_deduplicated_items = (
            await cls.__deduplicate_list_using_semantic_similarity(
                items_to_deduplicate, initial_semantic_threshold
            )
        )
        shuffled_items = semantic_deduplicated_items.copy()
        random.shuffle(shuffled_items)

        batch_size = 8  # Roughly what GPT can handle well
        batches = [
            shuffled_items[i : i + batch_size]
            for i in range(0, len(shuffled_items), batch_size)
        ]

        batch_tasks = [
            cls.__deduplicate_list_in_batch(batch, prompt_context) for batch in batches
        ]
        deduplicated_batches = await asyncio.gather(*batch_tasks)
        deduplicated_batches_flattened = [
            item for sublist in deduplicated_batches for item in sublist
        ]

        final_deduplicated_list = await cls.__deduplicate_list_in_batch(
            deduplicated_batches_flattened, prompt_context
        )
        cls.__log_deduplication_results(items_to_deduplicate, final_deduplicated_list)
        return final_deduplicated_list

    @classmethod
    async def __deduplicate_list_in_batch(
        cls, items_to_deduplicate: list[str], prompt_context: str
    ) -> list[str]:
        deduplication_prompt = clean_indents(
            f"""
            # Instructions
            You are a research contractor helping a client deduplicate a list.

            Here is is the context to the project below in triple backticks:
            ```
            {prompt_context}
            ```

            Please deduplicate the list of items given to you, keeping the instructions in mind.
            Remove any redundant or duplicate entries while preserving unique information.
            Pick the most representative example, and copy its word verbatim.
            Return only the deduplicated list as a JSON array of strings, with no additional text or explanation.

            # Example
            Items to deduplicate:
            [
                "**Hiroshima**: Hiroshima was bombed in 1945 during World War II",
                "**Nagasaki (1945)**: Nagasaki was bombed in 1945 during World War II",
                "**Hiroshima**: In 1945, Hiroshima was bombed with a nuclear warhead. This was the first time a nuclear weapon was used in warfare",
                "**Nagasaki**: Nagasaki was bombed with an atomic bomb called Little Boy",
                "**Nagasaki Nuclear Attack**: The war with Japan ended after the United States dropped an atomic bomb on Nagasaki",
            ]
            Deduplicated list:
            [
                "**Hiroshima**: In 1945, Hiroshima was bombed with a nuclear warhead. This was the first time a nuclear weapon was used in warfare",
                "**Nagasaki (1945)**: Nagasaki was bombed in 1945 during World War II",
            ]

            # Your Turn
            Items to deduplicate:
            {items_to_deduplicate}

            Please deduplicate the list
            """
        )

        model = BasicLlm(temperature=0)
        deduplicated_items = await model.invoke_and_return_verified_type(
            deduplication_prompt, list[str]
        )

        logger.info(
            f"Deduplicated batch of size {len(items_to_deduplicate)} to {len(deduplicated_items)} items"
        )
        assert all(
            item in items_to_deduplicate for item in deduplicated_items
        ), "Items returned did not match original items"
        return deduplicated_items

    @classmethod
    async def deduplicate_list_one_item_at_a_time(
        cls,
        items: list[str],
        use_internet_search: bool = False,
        threshold_for_initial_semantic_check: float = 0.85,
    ) -> list[str]:
        deduplicated_items: list[str] = []
        for item in items:
            is_duplicate = await cls.determine_if_item_is_duplicate(
                item,
                deduplicated_items,
                use_internet_search,
                threshold_for_initial_semantic_check,
            )
            if not is_duplicate:
                deduplicated_items.append(item)
        cls.__log_deduplication_results(items, deduplicated_items)
        return deduplicated_items

    @classmethod
    async def determine_if_item_is_duplicate(
        cls,
        item: str,
        list_to_check: list[str],
        use_internet_search: bool = False,
        threshold_for_initial_semantic_check: float = 0.85,
    ) -> bool:
        if len(list_to_check) == 0:
            return False

        is_exact_duplicate = item in list_to_check
        if is_exact_duplicate:
            return True

        is_semantically_duplicate = cls.__determine_if_text_is_duplicate_semantically(
            item, list_to_check, threshold_for_initial_semantic_check
        )
        if is_semantically_duplicate:
            return True

        list_with_numbers = "\n".join(
            [f"({i}) {item}" for i, item in enumerate(list_to_check)]
        )
        deduplication_prompt = clean_indents(
            f"""
            If I were to add "{item}" to the following list, would it be a duplicate?
            {list_with_numbers}

            Please give your reasoning, and then answer with either "YES_IT_IS" or "NO_IT_IS_NOT"
            """
        )
        if use_internet_search:
            model = SmartSearcher(
                temperature=0, num_searches_to_run=2, num_sites_per_search=5
            )
        else:
            model = BasicLlm(temperature=0)

        is_duplicate = await model.invoke_and_check_for_boolean_keyword(
            deduplication_prompt,
            true_keyword="YES_IT_IS",
            false_keyword="NO_IT_IS_NOT",
        )
        if is_duplicate:
            logger.info(
                f"Determined item is a duplicate of another item in the list. Item: {item}"
            )
        return is_duplicate

    @classmethod
    async def __deduplicate_list_using_semantic_similarity(
        cls, items: list[str], threshold: float
    ) -> list[str]:
        deduplicated_items: list[str] = []
        for item in items:
            is_duplicate = cls.__determine_if_text_is_duplicate_semantically(
                item, deduplicated_items, threshold
            )
            if not is_duplicate:
                deduplicated_items.append(item)

        logger.info(
            f"Deduplicated {len(items)} items to {len(deduplicated_items)} items using semantic similarity"
        )
        return deduplicated_items

    @classmethod
    def __determine_if_text_is_duplicate_semantically(
        cls,
        text: str,
        list_to_compare_to: list[str],
        semantic_similarity_threshold: float,
    ) -> bool:
        """
        0.85 is good for an item like "1999 Moldovan referendum: description..."
        0.938 is good for a short item like "1999 Moldovan referendum"
        """
        texts_to_get_embeddings_for = [text] + list_to_compare_to
        try:
            embeddings = cls.__get_embeddings_using_huggingface(
                texts_to_get_embeddings_for
            )
        except Exception as e:
            logger.warning(
                f"Could not get embeddings using huggingface. Instead now getting embeddings with OpenAI. Error: {e}"
            )
            embeddings = cls.__get_embeddings_using_openai(texts_to_get_embeddings_for)

        text_embedding = embeddings[0]
        list_embeddings = embeddings[1:]

        for list_embedding in list_embeddings:
            similarity = cosine_similarity(
                np.array([text_embedding]), np.array([list_embedding])
            )[0][0]
            if similarity > semantic_similarity_threshold:
                return True
        return False

    @classmethod
    def __get_embeddings_using_openai(cls, texts: list[str]) -> list[list[float]]:
        # TODO: Track costs from this in llm cost tracker
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY is not set"
        client = OpenAI(api_key=api_key)

        def query(texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            return [embedding.embedding for embedding in response.data]

        return query(texts)

    @classmethod
    def __get_embeddings_using_huggingface(cls, texts: list[str]) -> list[list[float]]:
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        assert api_key is not None, "HUGGINGFACE_API_KEY is not set"
        headers = {"Authorization": f"Bearer {api_key}"}

        def query(texts: list[str]) -> list[list[float]]:
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            raise_for_status_with_additional_info(response)
            return response.json()

        return query(texts)

    @classmethod
    def __log_deduplication_results(
        cls, original_list: list[str], deduplicated_items: list[str]
    ) -> None:
        removed_items = [
            item for item in original_list if item not in deduplicated_items
        ]
        logger.info(f"Removed {len(removed_items)} duplicate items: {removed_items}")
        logger.info(
            f"Kept {len(deduplicated_items)} unique items: {deduplicated_items}"
        )
