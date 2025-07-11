from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_quick_search_high_context,
    query_asknews,
)
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
    agent_tool,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output

logger = logging.getLogger(__name__)


class DecomposedQuestion(BaseModel):
    question_or_idea_text: str
    resolution_process: str | None = None
    expected_resolution_date: datetime | str | None = None
    background_information: str | None = None
    past_resolution: str | None = None
    other_information: str | None = None


class DecompositionResult(BaseModel):
    general_research_and_approach: str
    decomposed_questions: list[DecomposedQuestion]

    @property
    def questions(self) -> list[str]:
        return [
            question.question_or_idea_text for question in self.decomposed_questions
        ]


class QuestionDecomposer:

    async def decompose_into_questions_deep(  # NOSONAR
        self,
        fuzzy_topic_or_question: str | None,
        related_research: str | None,
        additional_context: str | None,
        number_of_questions: int = 5,
        model: str | GeneralLlm = "openrouter/google/gemini-2.5-pro-preview",
    ) -> DecompositionResult:
        if isinstance(model, GeneralLlm):
            logger.warning(
                "You are using GeneralLlm for QuestionDecomposer. Converting to model name."
            )
        model_name: str = GeneralLlm.to_model_name(model)
        llm: AgentSdkLlm = AgentSdkLlm(model=model_name)
        instructions = clean_indents(
            f"""
            # Instructions
            You are a research assistant to a superforecaster helping both the superforecaster and his clients.

            You want to take an overarching topic or question they have given you and decompose it into a list of sub questions that that will lead to better understanding and forecasting the topic or question.

            Follow these instructions:
            1. Get general news on the topic
            2. Run follow up searches (either quick or via general news) to double click on some potential interesting topics that could lead to good questions
            3. Iteratively run searches (run multiple at a time if you can) until you feel ready to make questions. Don't go through more than 5 iterations on this.
            4. Come up with {number_of_questions} question ideas and run (in parallel) a quick search on each of them asking:
            > "How has [question] resolved in the past? Please provide a link to how to find this value or if you cannot find an exact resolution source, to suggest another question".
            5. Ask yourself if the question matches the criteria, and if you have a valid past resolution..
            6. Iterate until you have a list of high VOI questions that match all the criteria, and how they have resolved in the past. Feel free to change directions if you cannot find a resolution source. Don't go through more than 4 iterations on this step (run parallel searches on multiple questions at once to fit them all into 4 iterations)
            7. Pick your top questions
            8. Give your final answer in the requested format


            # Question requireemnts
            - The question hold shed light on the topic and have high VOI (Value of Information)
            - The question can be forecast and will be resolvable with public information
                - Good: "Will SpaceX launch a rocket on May 2nd 2023?"
                - Bad: "Will Elon mention his intention to launch a rocket on May 2nd 2023 in a private meeting?"
            - The question should be specific and not vague
            - The question should have a resolution date
            - Once the resolution date has passed, the question should be resolvable with 0.5-1.5hr of research
                - Bad: "Will a research paper in a established journal find that a new knee surgery technique reduces follow up surgery with significance by Dec 31 2023?" (To resolve this you have to do extensive research into all new research in a field)
                - Good: "Will public dataset X at URL Y show the number of follow ups to knee surgeries decrease by Z% by Dec 31 2023?" (requires only some math on a few data points at a known URL)
            - A good resolution source exists
                - Bad: "On 15 January 2026, will the general sentiment be generally positive for knee surgery professionals with at least 10 years of experience concerning ACL reconstruction research?" (There is no way to research this online. You would have to run a large study on knee professionals)
                - Good: "As of 15 January 2026, how many 'recruiting study' search results will there be on ClinicalTrials.gov when searching 'ACL reconstruction' in 'intervention/treatment'?" (requires only a search on a known website)
            - Don't forget to INCLUDE Links if you found any! Copy the links IN FULL to all your answers so others can know where you got your information.
            - The questions should match any additional criteria that the superforecaster/client has given you
            - The question should not be obvious. Consider the time range when determining this (short time ranges means things are less likely).
                - Bad: "Will country X start a war in the next 2 weeks" (Probably not, especially if they have not said anything about this)
                - Good: "Will country X start a war in the next year" (Could be possible, especially if there are risk factors)

            # Format
            Using markdown, you should give your response in the below format

            ```
            **General Research and Reasoning**:
            [Your background research, scratch pad notes, and explanation of your approach]

            **Question 1:**
            - Title: [Question Title]
            - Resolution process: [How you would resolve the question]
            - Expected resolution date: [Date you would resolve the question]
            - Background information: [Define terms and signficance that might not be obvious]
            - Past resolution: [How the question has resolved in the past with link]

            **Question 2:**
            - Title: [Question Title]
            - Resolution process: [How you would resolve the question]
            - Expected resolution date: [Date you would resolve the question]
            - Background information: [Define terms and signficance that might not be obvious]
            - Past resolution: [How the question has resolved in the past with link]
            ... etc ...
            ```

            Make sure to cite EVERYTHING! If you have a link, keep it (otherwise you plagerize)!
            Conversely, DO NOT share a link that you did not find in your own research!

            # Reiteration of your priorities
            The most important thing to get right is high VOI and high resolvability. Focus on these.

            # Things to consider
            - Anything that shed light on a good base rate (especially ones that already have data)
            - If there might be a index, established database, site, etc that would allow for a clear resolution
            - Major drivers of key metrics (what things cause large changes in key metrics?)
            - Today's date is {datetime.now().strftime("%Y-%m-%d")}
            - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")
            """
        )
        prompt = clean_indents(
            f"""
            # Your Task
            ## Topic/Question to Decompose
            Please decompose the following topic or question into a list of {number_of_questions} sub questions.

            Question/Topic: {fuzzy_topic_or_question}

            ## Additional Context/Criteria
            Here is some additional context/criteria that the superforecaster has mentioned:
            {additional_context}

            ## Related Research
            Here is some research that has already been done:
            {related_research}
            """
        )
        agent = AiAgent(
            name="Question Decomposer",
            instructions=instructions,
            model=llm,
            tools=[perplexity_quick_search_high_context, query_asknews],
            handoffs=[],
        )
        result = await AgentRunner.run(agent, prompt, max_turns=30)
        structured_output = await structure_output(
            str(result.final_output), DecompositionResult
        )
        return structured_output

    async def decompose_into_questions_fast(
        self,
        fuzzy_topic_or_question: str | None,
        related_research: str | None,
        additional_context: str | None,
        number_of_questions: int = 5,
        model: str | GeneralLlm = GeneralLlm.search_context_model(
            "openrouter/perplexity/sonar"
        ),
    ) -> DecompositionResult:
        llm = GeneralLlm.to_llm(model)

        prompt = clean_indents(
            f"""
            # Instructions
            You are a research assistant to a superforecaster

            You want to take an overarching topic or question they have given you and decompose
            it into a list of sub questions that that will lead to better understanding and forecasting
            the topic or question.

            Your research process should look like this:
            1. First get general news on the topic
            2. Then pick 3 things to follow up with. Search perplexity with these in parallel
            3. Then brainstorm 2x the number of question requested number of key questions requested
            4. Pick your top questions
            5. Give your final answer as:
                - Reasoning
                - Research Summary
                - List of Questions

            Don't forget to INCLUDE Links (including to each question if possible)!
            Copy the links IN FULL to all your answers so others can know where you got your information.

            # Question requirements
            - The question can be forecast and will be resolvable with public information
                - Good: "Will SpaceX launch a rocket in 2023?"
                - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
            - The question should be specific and not vague
            - The question should have an inferred date
            - The question should shed light on the topic and have high VOI (Value of Information)

            # Good candidates for follow up question to get context
            - Anything that shed light on a good base rate (especially ones that already have data)
            - If there might be a index, site, etc that would allow for a clear resolution
            - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")

            # Your Task
            ## Topic/Question to Decompose
            Please decompose the following topic or question into a list of {number_of_questions} sub questions.

            Question/Topic: {fuzzy_topic_or_question}

            ## Additional Context/Criteria
            {additional_context}

            ## Related Research
            {related_research}
            """
        )
        response = await llm.invoke(prompt)
        structured_output = await structure_output(response, DecompositionResult)
        return structured_output

    @agent_tool
    @staticmethod
    def decompose_into_questions_tool(
        fuzzy_topic_or_question: str,
        number_of_questions: int,
        additional_criteria_or_context_from_user: str | None = None,
        related_research: str | None = None,
    ) -> DecompositionResult:
        """
        Decompose a topic or question into a list of sub questions that helps to understand and forecast the topic or question. Can run in "fast" or "deep" mode.
        Use this when you need to create questions from a topic or question.

        Args:
            fuzzy_topic_or_question: The topic or question to decompose.
            number_of_questions: The number of questions to decompose the topic or question into. Default to 5.
            related_research: If you are running in fast mode, include as much research as possible to help make a good question (especially include important drivers/influencers of the topic). Otherwise set research to None by default
            additional_criteria_or_context_from_user: Additional criteria or context from the user (default to None)

        Returns:
            A DecompositionResult object with the following fields:
            - reasoning: The reasoning for the decomposition.
            - questions: A list of sub questions and metadata
        """
        if related_research:
            logger.debug(
                "You are running in deep mode but you have provided related research. This may lower the quality of the questions if this is provided by LLMs."
            )
        task = QuestionDecomposer().decompose_into_questions_deep(
            fuzzy_topic_or_question=fuzzy_topic_or_question,
            number_of_questions=number_of_questions,
            additional_context=additional_criteria_or_context_from_user,
            related_research=related_research,
        )
        return asyncio.run(task)


# V11: Works well with claude 3.7 thinking but not gpt-4.1
# instructions = clean_indents(
# f"""
# # Instructions
# You are a research assistant to a superforecaster helping both the superforecaster and his clients.

# You want to take an overarching topic or question they have given you and decompose it into a list of sub questions that that will lead to better understanding and forecasting the topic or question.

# Follow these instructions:
# 1. Get general news on the topic
# 2. Run follow up searches (either quick or via general news) to double click on some potential interesting topics that could lead to good questions
# 3. Iteraively run searches (run multiple at a time if you can) until you feel ready to make questions. Don't go through more than 2 iterations on this.
# 4. Come up with {number_of_questions} question ideas and run (in parallel) a quick search on each of them asking:
# > "How has [question] resolved in the past? Please provide a link to how to find this value or if you cannot find an exact resolution source, to suggest another question".
# 5. Ask yourself if the question matches the criteria, and if you have a valid past resolution..
# 6. Iterate until you have a list of high VOI questions that match all the criteria, and how they have resolved in the past. Feel free to change directions if you cannot find a resolution source. Don't go through more than 3 iterations on this step (run parallel searches on multiple questions at once to fit them all into 3 iterations)
# 7. Pick your top questions
# 8. Give your final answer in the requested format


# # Question requireemnts
# - The question hold shed light on the topic and have high VOI (Value of Information)
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket on May 2nd 2023?"
#     - Bad: "Will Elon mention his intention to launch a rocket on May 2nd 2023 in a private meeting?"
# - The question should be specific and not vague
# - The question should have a resolution date
# - Once the the resolution date has passed, the question should be resolvable with 0.5-1.5hr of research
#     - Bad: "Will a research paper in a established journal find that a new knee surgery technique reduces follow up surgery with significance by Dec 31 2023?" (To resolve this you have to do extensive research into all new research in a field)
#     - Good: "Will public dataset X at URL Y show the number of follow ups to knee surgeries decrease by Z% by Dec 31 2023?" (requires only some math on a few data points at a known URL)
# - A good resolution source exists
#     - Bad: "On 15 January 2026, will the general sentiment be generally positive for knee surgery professionals with at least 10 years of experience concerning ACL reconstruction research?" (There is no wasy to research this online. You would have to run a large study on knee professionals)
#     - Good: "As of 15 January 2026, how many 'recruiting study' search results will there be on ClinicalTrials.gov when searching 'ACL reconstruction' in 'intervention/treatment'?" (requires only a search on a known website)
# - Don't forget to INCLUDE Links if you found any! Copy the links IN FULL to all your answers so others can know where you got your information.
# - The questions should match any additional criteria that the superforecaster/client has given you
# - The question should not be obvious. Consider the time range when determining this (short time ranges means things are less likely).
#     - Bad: "Will country X start a war in the next 2 weeks" (Probably not, especially if they have not said anything about this)
#     - Good: "Will country X start a war in the next year" (Could be possible, especially if there are risk factors)

# # Format
# Using markdown, you should give your response in the below format

# ```
# **General Research and Reasoning**:
# [Your background research, scratch pad notes, and explanation of your approach]

# **Question 1:**
# - Title: [Question Title]
# - Resolution process: [How you would resolve the question]
# - Expected resolution date: [Date you would resolve the question]
# - Background information: [Define terms and signficance that might not be obvious]
# - Past resolution: [How the question has resolved in the past with link]

# **Question 2:**
# - Title: [Question Title]
# - Resolution process: [How you would resolve the question]
# - Expected resolution date: [Date you would resolve the question]
# - Background information: [Define terms and signficance that might not be obvious]
# - Past resolution: [How the question has resolved in the past with link]
# ... etc ...
# ```

# Make sure to cite everything! If you have a link, keep it (otherwise you plagerize)!

# # Reiteration of your priorities
# The most important thing to get right is high VOI and high resolvability. Focus on these.

# # Things to consider
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, established database, site, etc that would allow for a clear resolution
# - Major drivers of key metrics (what things cause large changes in key metrics?)
# - Today's date is {datetime.now().strftime("%Y-%m-%d")}
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")
# """
# )
# prompt = clean_indents(
# f"""
# # Your Task
# ## Topic/Question to Decompose
# Please decompose the following topic or question into a list of {number_of_questions} sub questions.

# Question/Topic: {fuzzy_topic_or_question}

# ## Additional Context/Criteria
# Here is some additional context/criteria that the superforecaster has mentioned:
# {additional_context}

# ## Related Research
# Here is some research that has already been done:
# {related_research}
# """
# )

# V10
# instructions = clean_indents(
# f"""
# # Instructions
# You are a research assistant to a superforecaster helping both the superforecaster and his clients.

# You want to take an overarching topic or question they have given you and decompose it into a list of sub questions that that will lead to better understanding and forecasting the topic or question.

# Follow these instructions:
# 1. Get general news on the topic
# 2. Run follow up searches to double click on some potential interesting topics that could lead to good questions
# 3. Iteraively run searches (run multiple at a time if you can) until you feel ready to make questions. Don't go through more than 2 iterations on this.
# 4. Come up with {number_of_questions} question ideas and run (in parallel) a quick search on each of them asking:
# > "How has [question] resolved in the past? Please provide a link to how to find this value or if you cannot find an exact resolution source, to suggest another question".
# 5. Ask yourself if the question matches the criteria, and if you have a valid past resolution..
# 6. Iterate until you have a list of high VOI questions that match all the criteria, and how they have resolved in the past. Feel free to change directions if you cannot find a resolution source. Don't go through more than 3 iterations on this step (run parallel searches on multiple questions at once to fit them all into 3 iterations)
# 7. Pick your top questions
# 8. Give your final answer in the requested format


# # Question requireemnts
# - The question hold shed light on the topic and have high VOI (Value of Information)
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket on May 2nd 2023?"
#     - Bad: "Will Elon mention his intention to launch a rocket on May 2nd 2023 in a private meeting?"
# - The question should be specific and not vague
# - The question should have a resolution date
# - Once the the resolution date has passed, the question should be resolvable with 0.5-1.5hr of research
#     - Bad: "Will a research paper in a established journal find that a new knee surgery technique reduces follow up surgery with significance by Dec 31 2023?" (To resolve this you have to do extensive research into all new research in a field)
#     - Good: "Will public dataset X at URL Y show the number of follow ups to knee surgeries decrease by Z% by Dec 31 2023?" (requires only some math on a few data points at a known URL)
# - A good resolution source exists
#     - Bad: "On 15 January 2026, will the general sentiment be generally positive for knee surgery professionals with at least 10 years of experience concerning ACL reconstruction research?" (There is no wasy to research this online. You would have to run a large study on knee professionals)
#     - Good: "As of 15 January 2026, how many 'recruiting study' search results will there be on ClinicalTrials.gov when searching 'ACL reconstruction' in 'intervention/treatment'?" (requires only a search on a known website)
# - Don't forget to INCLUDE Links if you found any! Copy the links IN FULL to all your answers so others can know where you got your information.
# - The questions should match any additional criteria that the superforecaster/client has given you
# - The question should not be obvious. Consider the time range when determining this (short time ranges means things are less likely).
#     - Bad: "Will country X start a war in the next 2 weeks" (Probably not, especially if they have not said anything about this)
#     - Good: "Will country X start a war in the next year" (Could be possible, especially if there are risk factors)

# # Format
# You should give your response in the below format

# ```
# **General Research and Reasoning**:
# [Your background research, scratch pad notes, and explanation of your approach]

# **Question 1:**
# - Title: [Question Title]
# - Resolution process: [How you would resolve the question]
# - Expected resolution date: [Date you would resolve the question]
# - Background information: [Define terms and signficance that might not be obvious]
# - Past resolution: [How the question has resolved in the past with link]

# **Question 2:**
# - Title: [Question Title]
# - Resolution process: [How you would resolve the question]
# - Expected resolution date: [Date you would resolve the question]
# - Background information: [Define terms and signficance that might not be obvious]
# - Past resolution: [How the question has resolved in the past with link]
# ... etc ...
# ```

# # Reiteration of your priorities
# The most important thing to get right is high VOI and high resolvability. Focus on these.

# # Things to consider
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, established database, site, etc that would allow for a clear resolution
# - Major drivers of key metrics (what things cause large changes in key metrics?)
# - Today's date is {datetime.now().strftime("%Y-%m-%d")}
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")
# """
# )
# prompt = clean_indents(
# f"""
# # Your Task
# ## Topic/Question to Decompose
# Please decompose the following topic or question into a list of {number_of_questions} sub questions.

# Question/Topic: {fuzzy_topic_or_question}

# ## Additional Context/Criteria
# Here is some additional context/criteria that the superforecaster has mentioned:
# {additional_context}

# ## Related Research
# Here is some research that has already been done:
# {related_research}
# """
# )


# v8
# instructions = clean_indents(
# f"""
# # Instructions
# You are a research assistant to a superforecaster helping both the superforecaster and his clients.

# You want to take an overarching topic or question they have given you and decompose it into a list of sub questions that that will lead to better understanding and forecasting the topic or question.

# Follow these instructions:
# 1. Get general news on the topic
# 2. Come up with {number_of_questions} question ideas and run (in parallel) a quick search on each of them asking:
# > "How has [question] resolved in the past? Please provide a link to how to find this value or if you cannot find an exact resolution source, to suggest another question".
# 3. Ask yourself if the question matches the criteria, and if you have a valid past resolution..
# 4. Iterate until you have a list of high VOI questions that match all the criteria, and how they have resolved in the past. Feel free to change directions if you cannot find a resolution source.
# 5. Pick your top questions
# 6. Give your final answer in the requested format

# # Question requireemnts
# - The question hold shed light on the topic and have high VOI (Value of Information)
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket on May 2nd 2023?"
#     - Bad: "Will Elon mention his intention to launch a rocket on May 2nd 2023 in a private meeting?"
# - The question should be specific and not vague
# - The question should have a resolution date
# - Once the the resolution date has passed, the question should be resolvable with 0.5-1.5hr of research
#     - Bad: "Will a research paper in a established journal find that a new knee surgery technique reduces follow up surgery with significance by Dec 31 2023?" (To resolve this you have to do extensive research into all new research in a field)
#     - Good: "Will public dataset X at URL Y show the number of follow ups to knee surgeries decrease by Z% by Dec 31 2023?" (requires only some math on a few data points at a known URL)
# - A good resolution source exists
#     - Bad: "On 15 January 2026, will the general sentiment be generally positive for knee surgery professionals with at least 10 years of experience concerning ACL reconstruction research?" (There is no wasy to research this online. You would have to run a large study on knee professionals)
#     - Good: "As of 15 January 2026, how many 'recruiting study' search results will there be on ClinicalTrials.gov when searching 'ACL reconstruction' in 'intervention/treatment'?" (requires only a search on a known website)
# - Don't forget to INCLUDE Links if you found any! Copy the links IN FULL to all your answers so others can know where you got your information.
# - The questions should match any additional criteria that the superforecaster/client has given you

# # Format
# You should give your response in the below format

# ```
# **General Research and Reasoning**:
# [Your background research, scratch pad notes, and explanation of your approach]

# **Question 1:**
# - Title: [Question Title]
# - Resolution process: [How you would resolve the question]
# - Expected resolution date: [Date you would resolve the question]
# - Background information: [Define terms and signficance that might not be obvious]
# - Past resolution: [How the question has resolved in the past with link]

# **Question 2:**
# - Title: [Question Title]
# - Resolution process: [How you would resolve the question]
# - Expected resolution date: [Date you would resolve the question]
# - Background information: [Define terms and signficance that might not be obvious]
# - Past resolution: [How the question has resolved in the past with link]
# ... etc ...
# ```

# # Reiteration of your priorities
# The most important thing to get right is high VOI and high resolvability. Focus on these.

# # Things to consider
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, established database, site, etc that would allow for a clear resolution
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")
# """
# )
# prompt = clean_indents(
# f"""
# # Your Task
# ## Topic/Question to Decompose
# Please decompose the following topic or question into a list of {number_of_questions} sub questions.

# Question/Topic: {fuzzy_topic_or_question}

# ## Additional Context/Criteria
# Here is some additional context/criteria that the superforecaster has mentioned:
# {additional_context}

# ## Related Research
# Here is some research that has already been done:
# {related_research}
# """
# )

# prompt_v7 = clean_indents(
#     f"""
#     # Instructions
#     You are a research assistant to a superforecaster helping both the superforecaster and his clients.
#     - You want to take an overarching topic or question they have given you and decompose it into a list of areas/themes/ideas that could help create sub questions that that will lead to better understanding and forecasting the topic or question.
#     - Focus on areas or themes that could have high VOI (Value of Information)
#     - Mention why this area/theme/idea could have high VOI when explored
#     - Mention your answer in the following format

#     # Format
#     **Reasoning**:
#     [Your background research, scratch pad notes, and explanation of your approach]

#     **Areas/Themes/Ideas**:
#     - [Area/Theme/Idea 1]: [1 paragraph explanation of why this area/theme/idea could have high VOI when explored]
#     - [Area/Theme/Idea 2]: [1 paragraph explanation of why this area/theme/idea could have high VOI when explored]
#     - [Area/Theme/Idea 3]: [1 paragraph explanation of why this area/theme/idea could have high VOI when explored]
#     - ...

#     # Your Task
#     ## Topic/Question to Decompose
#     Please decompose the following topic or question into a list of {number_of_questions} areas/themes/ideas.

#     Thistion/Topic: {fuzzy_topic_or_question}

#     ## Additional Context/Criteria
#     Here is some additional context/criteria that the superforecaster has mentioned:
#     {additional_context}

#     ## Related Research
#     Here is some research that has already been done:
#     {related_research}
#     """
# )


# Problems: Makes up resolution sources (might be useful otherwise)
# prompt_v6 = clean_indents(
# f"""
# # Instructions
# You are a research assistant to a superforecaster helping both the superforecaster and his clients.

# You want to take an overarching topic or question they have given you and decompose it into a list of sub questions that that will lead to better understanding and forecasting the topic or question.

# Your research process should look like this:
# 1. First get general up to date news and background info on the topic (what is the topic? What is current)
# 2. Brainstorm and search 3 follow up questions
# 3. Then brainstorm 2x the number of question requested number of key questions requested
# 4. Pick your top questions
# 5. Give your final answer in the requested format

# # Question requireemnts
# - The question should shed light on the topic and have high VOI (Value of Information)
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket on May 2nd 2023?"
#     - Bad: "Will Elon mention his intention to launch a rocket on May 2nd 2023 in a private meeting?"
# - The question should be specific and not vague
# - The question should have a resolution date
# - Once the the resolution date has passed, the question should be resolvable with 0.5-1.5hr of research
#     - Bad: "Will a research paper in a established journal find that a new knee surgery technique reduces follow up surgery with significance by Dec 31 2023?" (To resolve this you have to do extensive research into all new research in a field)
#     - Good: "Will public dataset X at URL Y show the number of follow ups to knee surgeries decrease by Z% by Dec 31 2023?" (requires only some math on a few data points at a known URL)
# - A good resolution source exists
#     - Bad: "On 15 January 2026, will the general sentiment be generally positive for knee surgery professionals with at least 10 years of experience concerning ACL reconstruction research?" (There is no wasy to research this online. You would have to run a large study on knee professionals)
#     - Good: "As of 15 January 2026, how many 'recruiting study' search results will there be on ClinicalTrials.gov when searching 'ACL reconstruction' in 'intervention/treatment'?" (requires only a search on a known website)
# - Don't forget to INCLUDE Links if you found any! Copy the links IN FULL to all your answers so others can know where you got your information.
# - The questions should match any additional criteria that the superforecaster/client has given you

# # Format
# You should give your response in the below format

# ```
# **General Research and Reasoning**:
# [Your background research, scratch pad notes, and explanation of your approach]

# **Question 1:**
# - Title: [Question Title]
# - Resolution process: [How you would resolve the question]
# - Expected resolution date: [Date you would resolve the question]
# - Why this is a useful question: [Reasoning]

# **Question 2:**
# - Title: [Question Title]
# - Resolution process: [How you would resolve the question]
# - Expected resolution date: [Date you would resolve the question]
# - Why this is a useful question: [Reasoning]

# ... etc ...
# ```

# # Reiteration of your priorities
# The most important thing to get right is high VOI and high resolvability. Focus on these.

# # Things to consider
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, established database, site, etc that would allow for a clear resolution
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")


# # Your Task
# ## Topic/Question to Decompose
# Please decompose the following topic or question into a list of {number_of_questions} sub questions.

# Question/Topic: {fuzzy_topic_or_question}

# ## Additional Context/Criteria
# Here is some additional context/criteria that the superforecaster has mentioned:
# {additional_context}

# ## Related Research
# Here is some research that has already been done:
# {related_research}
# """
# )


# prompt_v5 = f"""
# # Instructions
# You are a research assistant to a superforecaster

# You want to take an overarching topic or question they have given you and decompose
# it into a list of sub questions that that will lead to better understanding and forecasting
# the topic or question.

# Your research process should look like this:
# 1. First get general news on the topic
# 2. Then pick 3 things to follow up with. Search perplexity with these in parallel
# 3. Then brainstorm 2x the number of question requested number of key questions requested
# 4. Pick your top questions
# 5. Give your final answer as:
#     - Reasoning
#     - Research Summary
#     - List of Questions

# Don't forget to INCLUDE Links (including to each question if possible)!
# Copy the links IN FULL to all your answers so others can know where you got your information.

# # Question requireemnts
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket in 2023?"
#     - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
# - The question should be specific and not vague
# - The question should have an inferred date
# - The question should shed light on the topic and have high VOI (Value of Information)

# # Good candidates for follow up question to get context
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, site, etc that would allow for a clear resolution
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")


# # Your Task
# ## Topic/Question to Decompose
# Please decompose the following topic or question into a list of {number_of_questions} sub questions.

# Question/Topic: {fuzzy_topic_or_question}

# ## Additional Context/Criteria
# {additional_context}

# ## Related Research
# {related_research}
# """


# agent_instructions_v4 = """
# # Instructions
# You are a research assistant to a superforecaster

# You want to take an overarching topic or question they have given you and decompose
# it into a list of sub questions that that will lead to better understanding and forecasting
# the topic or question.

# Your research process should look like this:
# 1. First get general news on the topic
# 2. Then pick 3 things to follow up with. Search perplexity with these in parallel
# 3. Then brainstorm 2x the number of question requested number of key questions requested
# 4. Pick your top questions
# 5. Give your final answer as:
#     - Reasoning
#     - Research Summary
#     - List of Questions

# Don't forget to INCLUDE Links (including to each question if possible)!
# Copy the links IN FULL to all your answers so others can know where you got your information.

# # Question requireemnts
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket in 2023?"
#     - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
# - The question should be specific and not vague
# - The question should have an inferred date
# - The question should shed light on the topic and have high VOI (Value of Information)

# # Good candidates for follow up question to get context
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, site, etc that would allow for a clear resolution
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")

# DO NOT ask follow up questions. Just execute the plan the best you can.
# """


# agent_instructions_v1 = """
# # Instructions
# You are a research assistant to a superforecaster

# You want to take an overarching topic or question they have given you and decompose
# it into a list of sub questions that that will lead to better understanding and forecasting
# the topic or question.

# Your research process should look like this:
# 1. First get general news on the topic and run a perplexity search
# 2. Pick 3 ideas things to follow up with and search perplexity with these
# 3. Then brainstorm 2x the number of question requested number of key questions requested
# 4. Pick  your top questions
# 5. Give your final answer as:
#     - Reasoning
#     - Research Summary
#     - List of Questions

# # Question requireemnts
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket in 2023?"
#     - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
# - The question should be specific and not vague
# - The question should have an inferred date
# - The question should shed light on the topic and have high VOI (Value of Information)

# # Good candidates for follow up question to get context
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, site, etc that would allow for a clear resolution

# DO NOT ask follow up questions. Just execute the plan the best you can.
# """


# agent_instructions_v2 = """
# # Instructions
# You are a research assistant to a superforecaster

# You want to take an overarching topic or question they have given you and decompose
# it into a list of sub questions that that will lead to better understanding and forecasting
# the topic or question.

# Your research process should look like this:
# 1. First get general news on the topic (run general news tool and perplexity search)
# 2. List out 5-20 of the major drivers of the topic
# 3. Pick your top questions based on VOI (Value of Information) for predicting the overarching topic
# 4. In a "FINAL ANSWER" section list out:
#     - 2 paragraph summary of the research
#     - Overarching Reasoning
#     - List of Questions you chose
#     - Dont forget to INCLUDE LINKS for everything!

# # Question requireemnts
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket in 2023?"
#     - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
# - The question should be specific and not vague
# - The question should have an inferred date
# - The question should shed light on the topic and have high VOI (Value of Information)

# # Good candidates for follow up question to get context
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, site, etc that would allow for a clear resolution
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")

# DO NOT ask follow up questions. Just execute the plan the best you can.
# """
