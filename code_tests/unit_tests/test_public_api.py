import forecasting_tools


def test_public_api_imports() -> None:
    """
    Tests that all expected public members are importable from the forecasting_tools package.
    This helps detect accidental removal or renaming of public API components.
    """
    public_members: list[str] = [
        "BaseRateResearcher",
        "Estimator",
        "FactCheckedItem",
        "NicheListResearcher",
        "KeyFactorsResearcher",
        "ScoredKeyFactor",
        "QuestionGenerator",
        "TopicGenerator",
        "clean_indents",
        "Claude35Sonnet",
        "DeepSeekR1",
        "Gpt4o",
        "Gpt4oVision",
        "Gpt4oMetaculusProxy",
        "Perplexity",
        "ExaSearcher",
        "GeneralLlm",
        "MonetaryCostManager",
        "BenchmarkForBot",
        "BinaryReport",
        "DataOrganizer",
        "ForecastReport",
        "ReasonedPrediction",
        "MultipleChoiceReport",
        "PredictedOption",
        "PredictedOptionList",
        "NumericDistribution",
        "NumericReport",
        "BinaryQuestion",
        "MetaculusQuestion",
        "MultipleChoiceQuestion",
        "NumericQuestion",
        "QuestionState",
        "ForecastBot",
        "Notepad",
        "MainBot",
        "Q1TemplateBot2025",
        "Q2TemplateBot2025",
        "Q3TemplateBot2024",
        "Q4TemplateBot2024",
        "UniformProbabilityBot",
        "TemplateBot",
        "AskNewsSearcher",
        "run_benchmark_streamlit_page",
        "Benchmarker",
        "ApiFilter",
        "MetaculusApi",
        "PredictionExtractor",
        "SmartSearcher",
    ]

    missing_members: list[str] = []
    for member_name in public_members:
        if not hasattr(forecasting_tools, member_name):
            missing_members.append(member_name)
        else:
            member = getattr(forecasting_tools, member_name)
            assert (
                member is not None
            ), f"{member_name} is None in forecasting_tools public API"

    assert not missing_members, (
        f"The following members are missing from the forecasting_tools public API: "
        f"{', '.join(missing_members)}"
    )

    # Optional: Check if __all__ is defined and if it matches the list.
    # This provides an additional layer of verification if you decide to use __all__.
    if hasattr(forecasting_tools, "__all__"):
        # Ensure all members in public_members are in __all__
        all_missing_from_public = set(public_members) - set(forecasting_tools.__all__)
        assert not all_missing_from_public, (
            f"Mismatch: Members in public_members list but not in __all__: "
            f"{', '.join(all_missing_from_public)}"
        )

        # Ensure all members in __all__ are in public_members (helps catch outdated public_members list)
        public_missing_from_all = set(forecasting_tools.__all__) - set(public_members)
        assert not public_missing_from_all, (
            f"Mismatch: Members in __all__ but not in public_members list (please update test): "
            f"{', '.join(public_missing_from_all)}"
        )
