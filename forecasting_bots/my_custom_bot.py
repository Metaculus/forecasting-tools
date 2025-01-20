from forecasting_tools import (
    BinaryQuestion,
    Gpt4oMetaculusProxy,
    ReasonedPrediction,
    TemplateBot,
)


class MyCustomBot(TemplateBot):
    """A simple custom bot that uses GPT-4 to make predictions"""

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = f"""
        Question: {question.question_text}
        Background: {question.background_info}
        Resolution Criteria: {question.resolution_criteria}

        Research findings: {research}

        Please analyze this question and provide:
        1. Key factors that could influence the outcome
        2. Analysis of the evidence
        3. Final probability estimate

        End your response with a line stating 'Probability: XX%' where XX is 0-100.
        """

        reasoning = await Gpt4oMetaculusProxy(temperature=0.2).invoke(prompt)
        prediction = self._extract_forecast_from_binary_rationale(
            reasoning, max_prediction=1, min_prediction=0
        )

        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
