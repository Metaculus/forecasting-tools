import asyncio
import logging

import pytest

from forecasting_tools.agents_and_tools.base_rates.niche_list_researcher import (
    FactCheckedItem,
    NicheListResearcher,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import Gpt4o
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "things_to_generate",
    [
        "US supreme court cases about abortion",
        "Metorites that hit the United States between 1776 and 2024",
        "Federal antitrust lawsuits against tech companies ongoing, or resolved that have started between Jan 1 1950 and Oct 16 2024",
    ],
)
def test_large_lists_fail(things_to_generate: str) -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            NicheListResearcher(things_to_generate).research_niche_reference_class()
        )


# TODO:
# Add questions for
# - A question with a high number of expected items
# - Question needing correct recent information
@pytest.mark.parametrize(
    "things_to_generate, expected_items",
    [
        (
            "Times the US has dropped a nuke on a country during war with that country between 1776 and 2024",
            [
                "Hiroshima",
                "Nagasaki",
            ],
        ),
        (
            "Times something other than an individual person have been the Time Magazine's Person of the Year between 2012 and 2021",
            [
                "2014: Ebola fighters",  # See https://www.today.com/popculture/time-person-of-the-year-rcna59809
                "2017: The Silence Breakers",  # See above
                "2018: The Guardians",  # See above
            ],
        ),
        (
            "Countries that have successfully achieved a soft landing on the moon before Dec 30 2023",
            [
                "Russia",  # https://www.ndtvprofit.com/chandrayaan-3/how-many-countries-have-successfully-soft-landed-on-moon-bqc
                "China",  # See above
                "United States",  # See above
                "India",  # https://www.bbc.com/news/world-asia-india-66594520
            ],
        ),
        (
            "Countries that have successfully achieved a soft landing on the moon before Jan 20 2024",
            [
                "Russia",
                "China",
                "United States",
                "India",
                "Japan",  # See https://en.wikipedia.org/wiki/Smart_Lander_for_Investigating_Moon#:~:text=making%20Japan%20the%20fifth%20country%20to%20soft%2Dland%20a%20spacecraft%20on%20the%20Moon
            ],
        ),
        (
            "Successful Coups in Senegal between 1960 and Jul 20 2024",
            [],  # No instances
        ),  # See https://en.wikipedia.org/wiki/Politics_of_Senegal#:~:text=Senegal%20is%20one%20of%20the%20few%20African%20states%20that%20has%20never%20experienced%20a%20coup%20d%27%C3%A9tat
        (
            "Times when it was reported that more than 1 Billion birds were in the sky over the USA before Oct 16 2024",
            [
                "6-7 October 2023"  # https://birdcast.info/news/the-first-billion-bird-night-6-7-october-2023/
            ],
        ),
        (
            "Times a Moldovan (not Transnistrian) referendum was voted on before Oct 30 2024",
            [
                "1994 Moldovan referendum",  # https://en.wikipedia.org/wiki/Elections_in_Moldova#Referendums:~:text=Referendums%5Bedit,2019%20Moldovan%20referendum
                "1999 Moldovan constitutional referendum",  # See above
                "2010 Moldovan constitutional referendum",  # See above
                "2019 Moldovan referendum",  # See above
                "2024 Moldovan European Union membership referendum",  # See above
                # Should I add this to the lists? -> "2014 Gagauzia Referendum on Customs Union",
                # https://www.rferl.org/a/moldova-gagauz-referendum-counting/25251251.html
                # Moldova said it was illegal, but it passed with 98% https://balkaninsight.com/2024/01/31/moldova-condemns-separatist-march-in-restive-gagauzia-region/
                # See Perplexity's attempt: https://www.perplexity.ai/search/list-for-me-every-moldovan-ref-qxcLgs2WQbSpW8HDIayNEg
            ],
        ),
        (
            "Times a Moldovan (not Transnistrian) referendum was voted on before Oct 16 2024",
            [
                "1994 Moldovan referendum",
                "1999 Moldovan constitutional referendum",
                "2010 Moldovan constitutional referendum",
                "2019 Moldovan referendum",
            ],
        ),  # See above
        (
            "Times a Moldovan (not Transnistrian) referendum got a majority of votes in favor of passing the referendum before Oct 16 2024",
            [
                "1994 Moldovan referendum",  # https://en.wikipedia.org/wiki/1994_Moldovan_referendum#:~:text=was%20approved%20by%2097.9%25%20of%20voters
                "1999 Moldovan constitutional referendum",  # https://en.wikipedia.org/wiki/1999_Moldovan_constitutional_referendum#:~:text=The%20proposal%20was%20approved%20by%2064%25%20of%20voters
                "2010 Moldovan constitutional referendum",  # https://en.wikipedia.org/wiki/2010_Moldovan_constitutional_referendum#:~:text=Of%20those%20who%20had%20cast%20their%20vote%2C%2087.83%25%20chose%20%22Yes%22.%5B2%5D%20However%2C%20the%20referendum%20did%20not%20pass%20because%20only%2030.29%25%20of%20voters%20turned%20out%2C%20short%20of%20the%20necessary%2033%25%20for%20the%20referendum%20to%20be%20considered%20valid.%5B2%5D
                "2019 Moldovan referendum",  # https://en.wikipedia.org/wiki/2019_Moldovan_referendum#:~:text=Results%5Bedit,Source%3A%20CEC
            ],
        ),
        (
            "Times Apple was successfully sued by another entity for patent violations before Oct 16 2024",
            # Consider asking instead whether Apple got money in the case as 'success' is not super clear and how do you define the difference between a string of court cases (Virtenx) versus not
            [
                "Creative Technology v. Apple, Inc. (menu structure)",  # Settled in 2006 https://en.wikipedia.org/wiki/Litigation_involving_Apple_Inc.#:~:text=In%20August%202006%2C%20Apple%20and%20Creative%20settled%20the%20suit%20with%20Apple%20agreeing%20to%20pay%20Creative%20%24100%20million%20USD%20for%20the%20right%20to%20implement%20Creative%27s%20method%20of%20sorting%20songs%20on%20the%20iPod
                "Apple vs. Masimo (watch features)",  # Ban on some watch features https://time.com/6692718/apple-watch-masimo-alivecor-patent-antitrust-legal-explainer/#:~:text=the%20ITC%20imposed,welcomed%20by%20Masimo.
                "Typhoon Touch Technologies (touch screen)",  # Settled in 2010 https://en.wikipedia.org/wiki/Litigation_involving_Apple_Inc.#:~:text=In%202010%2C%20Apple%20settled%20with%20Typhoon%20for%20an%20undisclosed%20sum%20and%20was%20then%20dismissed%20from%20the%20litigation%20as%20of%20September%202010
                "Nokia v. Apple (wireless, iPhone)",  # Settled in 2007 https://en.wikipedia.org/wiki/Litigation_involving_Apple_Inc.#:~:text=For%20an%20undisclosed%20amount%20of%20cash%20and%20future%20ongoing%20iPhone%20royalties%20to%20be%20paid%20by%20Apple%2C%20Nokia%20agreed%20to%20settle%2C%20with%20Apple%27s%20royalty%20payments%20retroactively%20back%2Dpayable%20to%20the%20iPhone%27s%20introduction%20in%202007%2C%20but%20with%20no%20broad%20cross%2Dlicensing%20agreement%20made%20between%20the%20companies
                "Science Applications International Corporation (Virnetx) vs Apple (2014)",  # Virnetx won in 2014 https://caselaw.findlaw.com/court/us-federal-circuit/1678303.html
                "Apple vs. VirnetX (2012)",
                "Apple vs. VirnetX (2016)",  # Did this one split into the 2017 and 2018 ones?
                "Apple vs. VirnetX (2017)",
                "Apple vs. VirnetX (2018)",
                "Apple vs. VirnetX (2019)",
                "Apple vs. VirnetX (2020)",  # Was this just a final trial of a previous case?
                # Court cases for  2012, 2016, 2017, 2018, 2020 # https://caldwellcc.com/case/virnetx-inc-v-apple-inc/
                # VirnetX got money in 2019 https://en.wikipedia.org/wiki/Litigation_involving_Apple_Inc.#:~:text=The%20first%20case,verdict%20against%20it
                # A later 2020 VirnetX case was unsuccessful https://www.macrumors.com/2024/02/20/apple-wins-virnetx-503-million/#:~:text=After%20Apple%20appealed%20the%20initial,on%20or%20license%20its%20patents.
                "Qualcomm vs. Apple (2017-2019): Power management/Download Speed",
                # Paid 31million https://www.qualcomm.com/news/releases/2019/03/qualcomm-wins-patent-infringement-case-against-apple-san-diego
                # https://www.reuters.com/article/technology/apple-infringed-three-qualcomm-patents-jury-finds-idUSKCN1QW2JZ/#:~:text=Mobile%20phone%20chip%20supplier%20Qualcomm%20Inc%20on%20Friday%20won%20a%20legal%20victory%20against%20iPhone%20maker%20Apple%20Inc%2C%20with%20a%20jury%20in%20federal%20court%20in%20San%20Diego%20finding%20that%20Apple%20owes%20Qualcomm%20about%20%2431%20million%20for%20infringing%20three%20of%20its%20patents.
                "Optis Wireless Technology vs. Apple (2019-2022)",
                # Payment and royalties https://caselaw.4ipcouncil.com/english-court-decisions/england-and-wales-high-court/optis-v-apple-high-court-justice#:~:text=In%20November%202021,payable.%5B15%5D
                "Smartflash LLC vs. Apple (2015)",  # Apple pas $533m https://arstechnica.com/tech-policy/2015/02/company-without-a-product-wins-533m-verdict-vs-apple-says-its-no-patent-troll/#:~:text=Smartflash%20LLC%20is,of%20its%20patents.
                "WiLAN vs. Apple (2018)",  # Paying $10m https://www.macrumors.com/2019/01/07/wilan-apple-dispute-damages-award-lowered/#:~:text=Back%20in%20August%2C%20a%20California,and%20reasonable%22%20settlement%20with%20Apple.
                "Ericsson vs. Apple",
                # Settled the dispute http://www.fosspatents.com/2022/12/ericsson-and-apple-settle-5g-patent.html#:~:text=Ericsson%20and%20Apple%20settle%205G%20patent%20dispute%20during%20Texas%20and%20ITC%20trials%2C%20agree%20on%20global%20cross%2Dlicense%3B%20%24400%20million%20one%2Doff%20payment%20plus%20ongoing%20royalties%20at%20unknown%20rate
                # Apple is defendant http://www.fosspatents.com/2022/09/ericsson-calls-apples-app-store.html#:~:text=Apple%20is%20being%20defended%20against,manual%20app%20review%20%28on%20average%29.
                "PanOptis vs. Apple",  # $506m and royalties https://appleinsider.com/articles/20/08/11/apple-ordered-to-pay-panoptis-5062m-for-infringing-lte-patents#:~:text=In%20Tuesday%27s%20decision%2C%20the%20jury,patents%20in%20suit%20were%20violated.
                "Core Wireless Licensing vs. Apple",
                # Original won the case for 7.3m https://www.patentlyapple.com/2018/08/a-federal-appeals-court-has-ruled-that-apple-didnt-infringe-one-of-two-patents-in-case-brought-on-by-core-wireless-licensing.html#:~:text=In%202016%20a,wherever%2C%20and%20whenever.%22
                # One claim was revered I think https://law.justia.com/cases/federal/appellate-courts/cafc/17-2102/17-2102-2018-08-16.html#:~:text=The%20court%20reversed%20in%20part%3B%20Core%E2%80%99s%20theory%20of%20infringement%20is%20inadequate%20to%20support%20a%20judgment%20on%20claim%2019.
                # Might be same case as Nokia v Apple https://www.patentlyapple.com/2018/08/a-federal-appeals-court-has-ruled-that-apple-didnt-infringe-one-of-two-patents-in-case-brought-on-by-core-wireless-licensing.html#:~:text=The%20verdict%20capped%20a%20trial%20that%20kicked%20off%20on%20Dec.%205%20centering%20on%20two%20patents%20that%20were%20Originaly%20owned%20by%20Nokia
                "Apple vs. Caltech (2016-2020)",  # Paid $838m https://appleinsider.com/articles/23/08/11/caltech-may-finally-settle-848-million-patent-case-against-apple?utm_medium=rss#:~:text=Caltech%20began%20its%20legal%20battle,amounts%20they%20had%20to%20pay.
                "Ironworks Patents LLC vs Apple Inc (2010)",  # Paid $10million. See filing 749 and filing 8 https://dockets.justia.com/docket/delaware/dedce/1:2010cv00258/43925
                "WARF vs Apple (2015)",  # $234mill https://news.wisc.edu/warf-wins-patent-infringement-lawsuit-against-apple/
                # Invalid
                # "Brazilian SEP Litigation",
                # There was a case ruled in Brazil in Ericcson's favor, but this was part of the larger Apple v Ericcson dispute
                # https://www.lickslegal.com/articles/ericsson-apple-settlement-came-hot-on-the-heels-of-landmark-brazilian-ruling-2#:~:text=In%20one%20of,in%20Ericsson%E2%80%99s%20favour.
                # http://www.fosspatents.com/2023/01/burgeoning-ip-and-antitrust.html#:~:text=But%20on%20a%20Sunday%20on,new%20agreement%20would%20be%20struck.
                # "Uniloc vs. Apple",
                # Case dismissed: https://casetext.com/case/uniloc-usa-inc-v-apple-inc-10#:~:text=Having%20earlier%20found,is%20DISMISSED.
                # Lacked standing https://ipwatchdog.com/2022/11/06/cafc-delivers-win-loss-uniloc-separate-precedential-rulings-standing/id=152576/#:~:text=Then%2C%20in%20December,the%20alleged%20infringer.%E2%80%9D
                # "Corephotonics v. Apple"
                # https://natlawreview.com/article/some-touch-needed-federal-circuit-partially-confirms-ptabs-view-analogous-art#:~:text=After%20Corephotonics%20sued,with%20other%20references).
                # https://cafc.uscourts.gov/opinions-orders/22-1350.OPINION.9-11-2023_2188207.pdf
                # https://cafc.uscourts.gov/opinions-orders/22-1340.OPINION.10-16-2023_2205991.pdf
                # "Second QualComm case"
                # Is this a different one? It was settled... I think Apple initiated? # https://www.reuters.com/technology/apple-loses-second-bid-challenge-qualcomm-patents-us-supreme-court-2022-10-03/#:~:text=The%20companies%20settled%20their%20underlying%20fight%20in%202019%2C%20signing%20an%20agreement%20worth%20billions%20of%20dollars%20that%20let%20Apple%20continue%20using%20Qualcomm%20chips%20in%20iPhones
                # "Qualcomm vs. Apple : Modems - Apple Sued Qualcomm" # They settled  https://www.inquartik.com/blog/case-intel-apple-qualcomm/#:~:text=Apple%20initially%20sued%20Qualcomm%20for%20%241%20billion%2C%20in%20China%2C%20it%20was%20for%20%24145%20million.%20During%20the%20period%20of%20legal%20action%2C%20Apple%20used%20Intel%E2%80%99s%20modems%20to%20build%20the%20iPhone%20XS.
                # List of other cases (valid ones have been processed) https://en.wikipedia.org/w/index.php?title=Litigation_involving_Apple_Inc.
            ],
        ),
        (
            "Times Apple settled a lawsuit regarding patent violations for which it was a defendant before Oct 16 2024",
            [
                "Creative Technology v. Apple, Inc. (menu structure)",
                "Typhoon Touch Technologies (touch screen)",
                "Nokia v. Apple (wireless, iPhone)",
                "Ericsson vs. Apple",
                # Consider if Perplexity got any that I missed (see second message) https://www.perplexity.ai/search/countries-that-have-successful-b2nwlS8DSeqIjBFsKm94og
            ],
        ),  # See above
        (
            "Times North Korea tested a nuclear weapon between Oct 1 and Dec 31 of an U.S. presidential election year for which this happened before Oct 16 2024",
            [
                "Obama 2008",  # https://beyondparallel.csis.org/no-major-activity-observed-at-punggye-ri-amid-increased-tension/#:~:text=CSIS%20studies%20have%20found%20a%20correlation%20between%20increased%20North%20Korean%20testing%20of%20missiles%20and%20nuclear%20devices%20and%20U.S.%20presidential%20and%20mid%2Dterm%20election%20years.%20Both%20the%20Trump%20and%20Obama%20administrations%2C%20for%20example%2C%20experienced%20North%20Korean%20nuclear%20tests%20shortly%20after%20being%20elected.
                "Trump 2016",  # see above
            ],
        ),
    ],
)
async def test_exhaustive_list_generation(
    things_to_generate: str, expected_items: list[str]
) -> None:
    with MonetaryCostManager() as cost_manager:
        all_items_including_incorrect = asyncio.run(
            NicheListResearcher(
                type_of_thing_to_generate=things_to_generate
            ).research_niche_reference_class(return_invalid_items=True)
        )
    list_markdown = FactCheckedItem.make_markdown_with_valid_and_invalid_lists(
        all_items_including_incorrect
    )
    exhaustive_list = [item for item in all_items_including_incorrect if item.is_valid]
    logger.info(
        f"\nCost: {cost_manager.current_usage}\nGenerated list: {list_markdown}\n\n Expected list: {expected_items}"
    )
    assert len(exhaustive_list) == len(expected_items)
    item_names = [item.item_name for item in exhaustive_list]
    assert len(set(item_names)) == len(item_names)
    for item in exhaustive_list:
        assert isinstance(item, FactCheckedItem)
        assert item.is_valid
        assert len(item.supporting_urls) > 0
        assert item.one_line_fact_check_summary

    model = Gpt4o(temperature=0)
    prompt = clean_indents(
        f"""
        You are a teacher evaluating the research of a student.

        You asked the student to find a list of "{things_to_generate}".

        Please analyze the generated items and determine if they match the expected items. Consider the following:
        1. Are all expected items present in the generated list?
        2. Are there any additional items in the generated list that weren't expected?
        3. Are the items equivalent in meaning, even if the wording is slightly different?

        Provide your assessment in the format below as a json object:
        {{
            "analysis": "Walk through step by step the criteria"
            "items_not_shared_between_lists": [
                "Items 1 not in both lists",
                "Items 2 not in both lists",
                ...
            ]
            "passing": true/false
        }}

        Expected items:
        {expected_items if len(expected_items) != 0 else "No expected items"}

        Generated items:
        {item_names if len(item_names) != 0 else "No generated items"}

        Please give the json object and nothing else
        """
    )
    response = await model.invoke_and_return_verified_type(prompt, dict)
    assert response[
        "passing"
    ], f"Generated list does not match expected items. Analysis: {response['analysis']}, Items not shared between lists: {response['items_not_shared_between_lists']}"

    logger.info(f"Final cost: {cost_manager.current_usage}")
    assert 0.01 < cost_manager.current_usage < 0.30


# NOTE: Started research on this item, and found there were more instances than expected (more than 25)

# My searches
# - Federal antitrust lawsuits against tech companies between Jan 1 1950 and Oct 16 2024
# - wikipedia federal antitrust lawsuits tech companies
# - Failed: Used quote from an article I found "On Monday a federal judge ordered Google to open up its Android app store to competition as punishment for maintaining an illegal monopoly in that market"
# - FTC legal library (search on site)
# - DOJ Antitrust case filings

# "Federal antitrust lawsuits against tech companies ongoing, or resolved that have started between Jan 1 1950 and Oct 16 2024",
# # General lists:
# # - Ongoing cases (7): https://www.economicliberties.us/tech-lawsuit-timelines/
# # - Wikipedia US antitrust lawsuits (~16/145): https://en.wikipedia.org/wiki/Category:United_States_antitrust_case_law
# # - DOJ list (<2419): https://www.justice.gov/atr/antitrust-case-filings
# # - FTC list (<5k): https://www.ftc.gov/legal-library
# # - Some big names (6/15): Does this give stats on how many cases have resulted in structural remedies
# # - FutureSearch https://app.futuresearch.ai/forecasts/eMocg/public
# [
#     "DOJ v. Google (filed 2020, remedies pending, declared monopolistic): Internet Search",
#         # https://www.economicliberties.us/dept-of-justice-v-google/#:~:text=The%20U.S.%20Department%20of%20Justice%2C%20along%20with%20a%20group%20of%20State%20Attorneys%20General%2C%20sued%20Google%20in%202020%20for%20illegally%20monopolizing%20the%20internet%20search%20engine%20and%20internet%20search%20advertising%20markets.
#         # https://www.theverge.com/23869483/us-v-google-search-antitrust-case-updates#:~:text=It%20has%20violated,and%20its%20products.
#     "DOJ v. Google (filed 2023, ongoing): Ad tech",
#         # https://www.economicliberties.us/dept-of-justice-v-google-adtech/
#     "FTC v. Facebook (filed 2020, ongoing): Social Networking",
#         # https://www.ftc.gov/legal-library/browse/cases-proceedings/191-0134-facebook-inc-ftc-v
#         # https://www.economicliberties.us/ftc-v-facebook/
#     "Texas et al. v. Google (filed 2020, ongoing): Advertising",
#         # https://www.economicliberties.us/texas-v-google/
#     "Colorado et al. v. Google (filed 2020, ongoing): Internet Search",
#         # Note: This was combined with DOJ's lawsuit
#         # https://www.economicliberties.us/colorado-v-google/
#     "Utah et al. v. Google (filed 2021, resolved 2023): Smartphone Application Market",
#         # Settled in September 2023
#         # Still to be approved?
#         # https://oag.ca.gov/news/press-releases/attorney-general-bonta-announces-700-million-settlement-google-monopolizing#:~:text=The%C2%A0settlement%C2%A0in,53%20attorneys%20general
#         # https://www.naag.org/multistate-case/utah-et-al-v-google-llc-no-321-cv-05227-n-d-cal-july-7-2021/
#         # https://www.economicliberties.us/utah-v-google/
#     "California v. Amazon (filed 2022, ongoing): Price Competition"
#         # Seems like the most recent progress is to just make sure the case happens: https://oag.ca.gov/news/press-releases/attorney-general-bonta-secures-court-decision-denying-amazon%E2%80%99s-attempt-evade
#     "-------- V Still to Validate Below V ----------",
#     "High-Tech Employee Antitrust Litigation",
#     "FTC v. Broadcom Inc. (2021)",
#     "FTC v. Meta Platforms, Inc. (2022)",
#     "U.S. v. Sabre Corp. (2019)",
#     "United States v. CDK Global, LLC (2018)",
#     "FTC v. Intel Corp. (1998)",
#     "U.S. v. Microsoft Corp. (1998)",
#     "U.S. v. Oracle Corporation (2004)",
#     "FTC v. Nvidia Corporation (2021)",
#     "United States v. Adobe Systems Inc., Apple Inc., Google Inc., Intel Corporation, Intuit Inc., and Pixar (2010)",
#     "U.S. v. eBay Inc. (2012)",
#     "United States v. Bazaarvoice, Inc. (2014)",
#     "FTC v. Qualcomm Inc. (2017)",
#     "U.S. v. IBM (1969)",
#     "United States v. Apple Inc. (2012)",
#     "? v. AT&T"
# ]
# (
#     "Times federal antitrust lawsuits against tech companies led to behavioral remedies between Jan 1 1950 and Oct 16 2024",
#     [
#         "Utah et al. v. Google (filed 2021, resolved 2023): Smartphone Application Market",
#     ]
# ),
# (
#     "Times federal antitrust lawsuits against tech companies led to structural remedies between Jan 1 1950 and Oct 16 2024",
#     [

#     ]
# ),
# (
#     "Federal antitrust lawsuits against tech companies starting in 2020",
#     [

#     ]
# )
