![PyPI version](https://badge.fury.io/py/forecasting-tools.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/forecasting-tools.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
[![Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/Dtq4JNdXnw)
[![PyPI Downloads](https://static.pepy.tech/badge/forecasting-tools/month)](https://pepy.tech/projects/forecasting-tools)
[![PyPI Downloads](https://static.pepy.tech/badge/forecasting-tools)](https://pepy.tech/projects/forecasting-tools)


# Quick Start
Install this package with `pip install forecasting-tools`

Demo website: https://forecasting-tools.streamlit.app/

Demo repo (get a Metaculus bot running in 30min): https://github.com/Metaculus/metac-bot-template

# Overview

This repository contains forecasting and research tools built with Python and Streamlit. The project aims to assist users in making predictions, conducting research, and analyzing data related to hard to answer questions (especially those from Metaculus).

Here are the tools most likely to be useful to you:
- 🎯 **Forecasting Bot:** General forecaster that integrates with the Metaculus AI benchmarking competition and provides a number of utilities. You can forecast with a pre-existing bot or override the class to customize your own (without redoing all the aggregation/API code, etc)
- 🔌 **Metaculus API Wrapper:** for interacting with questions and tournaments
- 🤖 **In-House Metaculus Bots**: You can see all the bots that Metaculus is running on their site in `run_bots.py`

Here are some other features of the project (not all are documented yet):
- **General LLM Wrapper:** A unified interface around litellm with retry logic, the Metaculus proxy, structured outputs, and cost tracking
- **Monetary Cost Manager:** for tracking AI and API expenses
- **Other experimental tools:** See the demo site for other AI forecasting tools that this project supports (not all are documented). Also see the `scripts` folder for other common workflows and entry points into the code.

All the examples below are in a Jupyter Notebook called `README.ipynb` which you can run locally to test the package (make sure to run the first cell though).

If you decide you want to join the Metaculus AI Benchmarking Tournament, it is recommended that you start [here](https://github.com/Metaculus/metac-bot-template). This repo is easy to start with and has a 30min tutorial for how to set it up.

Join the [discord](https://discord.gg/Dtq4JNdXnw) for updates and to give feedback (btw feedback is very appreciated, even just a quick "I did/didn't decide to use tool X for reason Y, though am busy and don't have time to elaborate" is helpful to know)

Note: This package is still in an experimental phase. The goal is to keep the package API fairly stable, though no guarantees are given at this phase especially with experimental tools. There will be special effort to keep the ForecastBot and TemplateBot APIs consistent.


# Forecasting Bot Building

## Using the Preexisting Bots

The package comes with two major pre-built bots:
- **MainBot**: The most accurate bot based on testing. May be more expensive.
- **TemplateBot**: A simple bot that's cheaper, easier to start with, and faster to run.

They both have roughly the same parameters. See below on how to use the TemplateBot to make forecasts.

### Forecasting on a Tournament


```python
from forecasting_tools import TemplateBot, MetaculusApi, GeneralLlm

# Initialize the bot
bot = TemplateBot(
    research_reports_per_question=3,  # Number of separate research attempts per question
    predictions_per_research_report=5,  # Number of predictions to make per research report
    publish_reports_to_metaculus=False,  # Whether to post the forecasts to Metaculus
    folder_to_save_reports_to="logs/forecasts/",  # Where to save detailed reports
    skip_previously_forecasted_questions=False,
    llms={ # LLM models to use for different tasks. Will use default llms if not specified. Requires the relevant provider environment variables to be set.
        "default": GeneralLlm(model="openrouter/google/gemini-2.5-pro", temperature=0),
        "summarizer": "openai/gpt-4o-mini",
    }
)

TOURNAMENT_ID = MetaculusApi.CURRENT_QUARTERLY_CUP_ID
reports = await bot.forecast_on_tournament(TOURNAMENT_ID)

# Print results (if the tournament is not active, no reports will be returned)
for report in reports:
    print(f"\nQuestion: {report.question.question_text}")
    print(f"Prediction: {report.prediction}")
```


    Question: Will a 2025 Major Atlantic Hurricane make landfall before September?
    Prediction: 0.35

    Question: Will China enact an export ban on a rare earth element to the United States before September 1, 2025?
    Prediction: 0.15

    Question: Will Israel strike the Iranian military in Iran again, before September 2025?
    Prediction: 0.28



### Forecasting Outside a Tournament


```python
from forecasting_tools import (
    TemplateBot,
    BinaryQuestion,
    MetaculusApi,
    DataOrganizer
)

# Initialize the bot
bot = TemplateBot(
    research_reports_per_question=3,
    predictions_per_research_report=5,
    publish_reports_to_metaculus=False,
)

# Get and forecast a specific question
question1 = MetaculusApi.get_question_by_url(
    "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
)
question2 = BinaryQuestion(
    question_text="Will YouTube be blocked in Russia?",
    background_info="...", # Or 'None'
    resolution_criteria="...", # Or 'None'
    fine_print="...", # Or 'None'
)

reports = await bot.forecast_questions([question1, question2])


# Print results
for report in reports:
    print(f"Question: {report.question.question_text}")
    print(f"Prediction: {report.prediction}")
    shortened_explanation = report.explanation.replace('\n', ' ')[:100]
    print(f"Reasoning: {shortened_explanation}...")

# You can also save and load questions and reports
file_path = "temp/reports.json"
DataOrganizer.save_reports_to_file_path(reports, file_path) # This will overwrite the file if it already exists
loaded_reports = DataOrganizer.load_reports_from_file_path(file_path)
```

    Question: Will humans go extinct before 2100?
    Prediction: 0.03
    Reasoning:  # SUMMARY *Question*: Will humans go extinct before 2100? *Final Prediction*: 3.0% *Total Cost*: $0...
    Question: Will YouTube be blocked in Russia?
    Prediction: 0.7
    Reasoning:  # SUMMARY *Question*: Will YouTube be blocked in Russia? *Final Prediction*: 70.0% *Total Cost*: $0...


The bot will:
1. Research the question
2. Generate multiple independent predictions
3. Combine these predictions into a final forecast
4. Save detailed research and reasoning to the specified folder
5. Optionally post the forecast to Metaculus (if `publish_reports_to_metaculus=True`)

Note: You'll need to have your environment variables set up (see the section below)

## Customizing the Bot
### General Customization
Generally all you have to do to make your own bot is inherit from the TemplateBot and override any combination of the 3 forecasting methods and the 1 research method. This saves you the headache of interacting with the Metaculus API, implementing aggregation of predictions, creating benchmarking interfaces, etc. Below is an example. It may also be helpful to look at the TemplateBot code (forecasting_tools/forecasting/forecast_bots/template_bot.py) for a more complete example.


```python
from forecasting_tools import (
    TemplateBot,
    MetaculusQuestion,
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    ReasonedPrediction,
    PredictedOptionList,
    NumericDistribution,
    SmartSearcher,
    MetaculusApi,
    GeneralLlm,
    PredictionExtractor
)
from forecasting_tools.util.misc import clean_indents

class MyCustomBot(TemplateBot):

    async def run_research(self, question: MetaculusQuestion) -> str:
        searcher = SmartSearcher(
            num_searches_to_run=2,
            num_sites_per_search=10
        )

        prompt = clean_indents(
            f"""
            Analyze this forecasting question:
            1. Filter for recent events in the past 6 months
            2. Don't include domains from youtube.com
            3. Look for current trends and data
            4. Find historical analogies and base rates

            Question: {question.question_text}

            Background Info: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            """
        )

        report = await searcher.invoke(prompt)
        return report

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = f"Please make a prediction on the following question: {question.question_text}. The last thing you write is your final answer as: 'Probability: ZZ%', 0-100"
        reasoning = await GeneralLlm(model="metaculus/gpt-4o", temperature=0).invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        ...

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        ...

custom_bot = MyCustomBot()
question = MetaculusApi.get_question_by_url(
    "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
)
report = await custom_bot.forecast_question(question)
print(f"Question: {report.question.question_text}")
print(f"Prediction: {report.prediction}")
print(f"Research: {report.research[:100]}...")
```

    Question: Will humans go extinct before 2100?
    Prediction: 0.05
    Research: # RESEARCH
    To analyze the question of whether humans will go extinct before 2100, we need to consider...


### Maintaining state with Notepad
If you want to maintain state between forecasts, you can use the `Notepad` object. This can be used to alternate forecasts between two different models, decide personalities for a bot up front, etc. There is a `Notepad` object made for every question forecasted.


```python
from forecasting_tools import (
    TemplateBot,
    BinaryQuestion,
    ReasonedPrediction,
    GeneralLlm,
    Notepad
)
import random

class NotepadBot(TemplateBot):

    async def _initialize_notepad(
        self, question: MetaculusQuestion
    ) -> Notepad:
        new_notepad = Notepad(question=question)
        random_personality = random.choice(["superforecaster", "financial analyst", "political advisor"])
        new_notepad.note_entries["personality"] = random_personality
        return new_notepad

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        notepad = await self._get_notepad(question)

        if notepad.total_predictions_attempted % 2 == 0:
            model = "openrouter/openai/gpt-4o"
        else:
            model = "anthropic/claude-3-5-sonnet-20240620"

        personality = notepad.note_entries["personality"]
        prompt = f"You are a {personality}. Forecast this question: {question.question_text}. The last thing you write is your final answer as: 'Probability: ZZ%', 0-100"
        reasoning = await GeneralLlm(model=model, temperature=0).invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
```

## Join the Metaculus tournament using this package
You can create your own custom bot through the package in your own repo. An example can be found [at this repo](https://github.com/Metaculus/metac-bot-template) which you can fork and edit.

## Setting Environment Variables
Whether running locally or through Github actions, you will need to set environment variables. All environment variables you might want are in `.env.template`. Generally you only need the METACULUS_TOKEN if running the Template. Having an EXA_API_KEY (see www.exa.ai) or PERPLEXITY_API_KEY (see www.perplexity.ai) is needed for searching the web. Make sure to put these variables in your `.env` file if running locally and in the Github actions secrets if running on Github actions.

# Important Utilities

## Metaculus API
The Metaculus API wrapper helps interact with Metaculus questions and tournaments. Grabbing questions returns a pydantic object, and supports important information for Binary, Multiple Choice, Numeric,and Date questions.


```python
from forecasting_tools import MetaculusApi, ApiFilter, DataOrganizer
from datetime import datetime


# Get a question by post id
question = MetaculusApi.get_question_by_post_id(578)
print(f"Question found with url: {question.page_url}")

# Get a question by url
question = MetaculusApi.get_question_by_url("https://www.metaculus.com/questions/578/human-extinction-by-2100/")
print(f"Question found with url: {question.page_url}")

# Get all open questions from a tournament
questions = MetaculusApi.get_all_open_questions_from_tournament(
    tournament_id=MetaculusApi.CURRENT_QUARTERLY_CUP_ID
)
print(f"Num tournament questions: {len(questions)}")

# Get questions matching a filter
api_filter = ApiFilter(
    num_forecasters_gte=40,
    close_time_gt=datetime(2023, 12, 31),
    close_time_lt=datetime(2024, 12, 31),
    scheduled_resolve_time_lt=datetime(2024, 12, 31),
    allowed_types=["binary", "multiple_choice", "numeric", "date"],
    allowed_statuses=["resolved"],
)
questions = await MetaculusApi.get_questions_matching_filter(
    api_filter=api_filter,
    num_questions=50, # Remove this field to make it not error if you don't get 50 questions. However it will only go through one page of questions which may miss questions matching the ApiFilter since some filters are handled locally.
    randomly_sample=False
)
print(f"Num filtered questions: {len(questions)}")

# Load and save questions/reports
file_path = "temp/questions.json"
DataOrganizer.save_questions_to_file_path(questions, file_path) # Will overwrite the file if it already exists
questions = DataOrganizer.load_questions_from_file_path(file_path)

# Get benchmark questions
benchmark_questions = MetaculusApi.get_benchmark_questions(
    num_of_questions_to_return=20
)
print(f"Num benchmark questions: {len(benchmark_questions)}")

# Post a prediction
MetaculusApi.post_binary_question_prediction(
    question_id=578, # Note that the question ID is not always the same as the post ID
    prediction_in_decimal=0.012  # Must be between 0.01 and 0.99
)
print("Posted prediction")

# Post a comment
MetaculusApi.post_question_comment(
    post_id=578,
    comment_text="Here's example reasoning for testing... This will be a private comment..."
)
print("Posted comment")
```

    Question found with url: https://www.metaculus.com/questions/578
    Question found with url: https://www.metaculus.com/questions/578
    Num tournament questions: 11
    Num filtered questions: 50
    Num benchmark questions: 20
    Posted prediction
    Posted comment


### Group Questions

Several of the methods above accept a `group_question_mode` parameter that controls how Metaculus group questions (e.g. "How many people will die of coronavirus in [period]?") are handled:
- `"exclude"` — drop group questions from the result.
- `"unpack_subquestions"` — turn each subquestion into a separate normal question.

For backwards compatibility, the default is `"exclude"` for `get_question_by_post_id`, `get_question_by_url`, `ApiFilter` (used by `get_questions_matching_filter`), and `get_benchmark_questions` — so group questions don't get overweighted in benchmarks. The exception is `get_all_open_questions_from_tournament`, which defaults to `"unpack_subquestions"` so all subquestions are forecasted as normal questions.

```python
from forecasting_tools import MetaculusApi, ApiFilter

# Unpack a group question into its subquestions
result = MetaculusApi.get_question_by_post_id(
    post_id=...,  # a group-question post
    group_question_mode="unpack_subquestions",
)  # returns list[MetaculusQuestion] for group posts

# Same option on a filtered query
api_filter = ApiFilter(
    allowed_statuses=["open"],
    group_question_mode="unpack_subquestions",
)
questions = await MetaculusApi.get_questions_matching_filter(api_filter=api_filter)
```

## General LLM
The `GeneralLlm` class is a wrapper around around litellm's acompletion function that adds some functionality like retry logic, calling the metaculus proxy, and cost callback handling. Litellm supports every model, most every parameter, and acts as one interface for every provider. See the litellm's acompletion function for a full list of parameters. Not all models will support all parameters. Additionally the Metaculus proxy doesn't support all models.


```python
prompt = "What is the weather in Tokyo?"
result = await GeneralLlm(model="gpt-4o").invoke(prompt)
result = await GeneralLlm(model="claude-3-5-sonnet-20241022").invoke(prompt)
result = await GeneralLlm(model="metaculus/claude-3-5-sonnet-20241022").invoke(prompt) # Adding 'metaculus' Calls the Metaculus proxy
result = await GeneralLlm(model="gemini/gemini-pro").invoke(prompt)
result = await GeneralLlm(
    model="perplexity/sonar-pro",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n\n"],
    response_format=...
).invoke(prompt)
```

Additionally `GeneralLlm` provides some interesting structured response options. It will call the model a number of times until it gets the response type desired. The type validation works for any type including pydantic types like `list[BaseModel]` or nested types like `list[tuple[str,dict[int,float]]]`.


```python
from forecasting_tools import GeneralLlm
from pydantic import BaseModel

class President(BaseModel):
    name: str
    age_at_death: int
    biggest_accomplishment: str

model = GeneralLlm(model="gpt-4o")
pydantic_model_explanation = model.get_schema_format_instructions_for_pydantic_type(President)
prompt = clean_indents(f"""
    You are a historian helping with a presidential research project. Please answer the following question:
    Who is Abraham Lincoln?
    Please provide the information in the following format:
    {pydantic_model_explanation}
    """)

verified_president = await model.invoke_and_return_verified_type(
    prompt,
    President,
    allowed_invoke_tries_for_failed_output=2
)
regular_output = await model.invoke(prompt)

print(f"President is pydantic: {isinstance(verified_president, President)}")
print(f"President: {verified_president}")
print(f"Regular output: {regular_output}")

```

    President is pydantic: True
    President: name='Abraham Lincoln' age_at_death=56 biggest_accomplishment="Preserving the Union during the American Civil War and issuing the Emancipation Proclamation, which began the process of freedom for America's slaves."
    Regular output: ```json
    {
      "name": "Abraham Lincoln",
      "age_at_death": 56,
      "biggest_accomplishment": "Preserving the Union during the American Civil War and issuing the Emancipation Proclamation, which began the process of freedom for America's slaves."
    }
    ```


You can also do some basic code execution.


```python
from forecasting_tools import GeneralLlm, clean_indents

model = GeneralLlm(model="gpt-4o")

code = clean_indents("""
    Please run a fermi estimate for the number of books published worldwide each year.
    Generate only code and give your reasoning in comments. Do not include any other text, only code.
    Assign your answer to the variable 'final_result'.
    """)

result, code = await model.invoke_and_unsafely_run_and_return_generated_code(
    code,
    expected_output_type=float,
    allowed_invoke_tries_for_failed_output=2
)

print(f"Result: {result}")
print(f"Code:\n{code}")
```

    Result: 10010000.0
    Code:
    # Fermi estimate for the number of books published worldwide each year

    # Estimate the number of countries in the world
    num_countries = 200  # Rough estimate of the number of countries

    # Estimate the number of books published per country per year
    # Assume a small country might publish around 100 books per year
    # Assume a large country might publish around 100,000 books per year
    # Use an average of these estimates for a rough calculation
    avg_books_per_country = (100 + 100000) / 2

    # Calculate the total number of books published worldwide each year
    total_books_worldwide = num_countries * avg_books_per_country

    # Assign the result to the variable 'final_result'
    final_result = total_books_worldwide


## Monetary Cost Manager
The `MonetaryCostManager` helps to track AI and API costs. It tracks expenses and errors if it goes over the limit. Leave the limit empty to disable the limit. Any call made within the context of the manager will be logged in the manager. CostManagers are async safe and can nest inside of each other. Please note that cost is only tracked after a call finishes, so if you concurrently batch 1000 calls, they will all go through even if the cost is exceeded during the middle of the execution of the batch. Additionally not all models support cost tracking. You will get a logger warning if your model is not supported. Litellm models with weird non-token pricing structures may be inaccurate as well.


```python
from forecasting_tools import MonetaryCostManager
from forecasting_tools import (
    ExaSearcher, GeneralLlm
)

max_cost = 5.00

with MonetaryCostManager(max_cost) as cost_manager:
    prompt = "What is the weather in Tokyo?"
    result = await GeneralLlm(model="gpt-4o").invoke(prompt)
    result = await ExaSearcher().invoke(prompt)
    # ... etc ...

    current_cost = cost_manager.current_usage
    print(f"Current cost: ${current_cost:.2f}")
```

# Local Development

## Environment Variables
The environment variables you need can be found in ```.env.template```. Copy this template as ```.env``` and fill it in.

## Docker Dev Container
Dev containers are reliable ways to make sure environments work on everyone's machine the first try and so you don't have to spend hours setting up your environment (especially if you have Docker already installed). If you would rather just use poetry, without the dev container, you can skip to "Alternatives to Docker". Otherwise, to get your development environment up and running, you need to have Docker Engine installed and running. Once you do, you can use the VSCode dev container pop-up to automatically set up everything for you.

### Install Docker
For Windows and Mac, you will download Docker Desktop. For Linux, you will download Docker Engine. (NOTE: These instructions might be outdated).

First download and setup Docker Engine using the instructions at the link below for your OS:
 * Windows: [windows-install](https://docs.docker.com/desktop/install/windows-install/)
 * Mac: [mac-install](https://docs.docker.com/desktop/install/mac-install/)
 * Linux: [install](https://docs.docker.com/engine/install/)
    * Do not install Docker Desktop for Linux, rather, select your Linux distribution on the left sidebar and follow the distribution specific instructions for Docker engine. Docker Desktop runs with a different environment in Linux.
    * Remember to follow the post-installation steps for Linux: [linux-postinstall](https://docs.docker.com/engine/install/linux-postinstall/)


### Starting the container
Once Docker is installed, when you open up the project folder in VSCode, you will see a pop up noting that you have a setup for a dev container, and asking if you would like to open the folder in a container. You will want to click "open in container". This will automatically set up everything you need and bring you into the container. If it doesn't show up, press `ctrl+shift+P` and type in `devcontainers: build and reopen in remote container`

If the Docker process times out in the middle of installing python packages you can run the `.devcontiner/postinstall.sh` manually. You may need to have the VSCode Docker extension and/or devcontainer extension downloaded in order for the pop up to appear.

Once you are in the container, poetry should have already installed a virtual environment. For VSCode features to use this environment, you will need to select the correct python interpreter. You can do this by pressing `Ctrl + Shift + P` and then typing `Python: Select Interpreter`. Then select the interpreter that starts with `.venv`.

A number of vscode extensions are installed automatically (e.g. linting). You may need to wait a little while and then reload the window (once for things to install, and a second for needed reload after installation). You can install personal vscode extensions in the dev environment.


### Managing Docker
There are many ways to manager Docker containers, but generally if you download the vscode Docker extension, you will be able to stop/start/remove all containers and images.


### Alternatives to Docker
If you choose not to run Docker, you can use poetry to set up a local virtual environment. If you are on Ubuntu, you should be able to just read through and then run `.devcontainer/postinstall.sh`. If you aren't on Ubuntu, check out the links in the postinstall file for where install instructions for dependencies were originally found. You may also want to take a look at VSCode extensions that would be installed (see the list in the `.devcontainer/devcontainer.json` file) so that some VSCode workplace settings work out of the box (e.g. automatic Black Formatting).


## Running the Front End
You can run any front end folder in the front_end directory by executing `streamlit run front_end/main.py`. This will start a development server for you that you can run. Streamlit makes it very easy to publish demos.

## Testing
This repository uses pytest tests are subdivided into folders 'unit_tests', 'integration'. Unit tests should always pass. You can run `pytest code_tests/unit_tests` or just `pytest` to run all of these

Also it's helpful to use the log file that is automatically placed at `logs/latest.log`

# Contributing

## Getting Started

1. **Fork the Repository**: Fork the repository on GitHub. Clone your fork locally: `git clone git@github.com:your-username/forecasting-tools.git`
2. **Set Up Development Environment**: Follow the "Local Development" section in the README to set up your environment
3. **Come up with an improvement**: Decide on something worth changing. Perhaps, you want to add your own custom bot to the forecasting_bots folder. Perhaps you want to add a tool that you think others could benefit from. As long as your code is clean, you can probably expect your contribution to be accepted, though if you are worried about adoption, feel free to chat on our discord or create an issue.
4. **Make a pull request**:
   - Make changes
   - Push your changes to your fork
   - Make sure you rebase with the upstream main branch before doing a PR (`git fetch upstream` and `git rebase upstream/main`)
   - Go to your fork in github, and choose the branch that you have that has your changes
   - You should see a 'Contribute' button. Click this and make a pull request.
   - Fill out the pull request template with a  description of what changed and why and Url for related issues
   - Request review from maintainers
   - Respond to any feedback and make requested changes

## Development Guidelines

1. **Code Style**
   - Code is automatically formatted on commit using precommit (this should be automatically installed when you start the devcontainer). Formatting is done using Black, Ruff, and some other tools.
   - Use type hints for all function parameters and return values
   - Use descriptive variable names over comments
   - Follow existing patterns in the codebase

2. **Testing**
   - Add tests where appropriate for new functionality. We aren't shooting for full code coverage, but you shouldn't make none.
   - Run unit tests locally before merging to check if you broke anything. See the 'Testing' section.
   - Integration tests often ping live LLM APIs and so will incur a non negligible cost but moderate cost. If you are able please run these as well.

## Questions or Issues?

- Join our [Discord](https://discord.gg/Dtq4JNdXnw) for questions
- Open an issue for bugs or feature requests

Thank you for helping improve forecasting-tools!
