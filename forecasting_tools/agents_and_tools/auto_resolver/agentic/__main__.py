import forecasting_tools
from forecasting_tools.agents_and_tools.auto_resolver.agentic import AgenticResolver
from forecasting_tools.agents_and_tools.auto_resolver.agentic import logger as target_logger
import asyncio
from dotenv import load_dotenv

load_dotenv()


from forecasting_tools.agents_and_tools.auto_resolver.assess import (
    ResolutionAssesser,
)
from forecasting_tools import MetaculusClient, ApiFilter
from dotenv import load_dotenv
import random

import logging


async def main():
    logging.getLogger(target_logger.name).setLevel(logging.INFO)
    
    # Fetch all resolved binary questions from AIB Fall 2025
    client = MetaculusClient()
    filter = ApiFilter(
        allowed_tournaments=[MetaculusClient.AIB_FALL_2025_ID],
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        group_question_mode="exclude",
        order_by="-published_time"
    )
    
    all_questions = await client.get_questions_matching_filter(filter)
    
    # Randomly sample questions
    sample_size = min(20, len(all_questions))
    sampled_questions = random.sample(all_questions, sample_size)
    question_ids = [q.id_of_post for q in sampled_questions if q.id_of_post is not None]
    
    
    # Create resolver
    resolver = AgenticResolver()

    # Create assessor with specific question IDs
    assesser = ResolutionAssesser(
        resolver, allowed_types=["binary"], questions=question_ids
    )

   
    # Run assessment
    report = await assesser.assess_resolver()

    # Print results
    print("\n" + "=" * 60)
    print("OpenAI Resolver Assessment Results")
    print(f"Tested on {len(question_ids)} random questions from AIB Fall 2025")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # Save detailed report to reports directory
    try:
        report_path = report.write_to_file(directory="reports")
        print(f"\nDetailed report saved to: {report_path}")
    except Exception as e:
        print(f"\nWarning: Could not save report to reports directory: {e}")

if __name__ == "__main__":
    asyncio.run(main())
