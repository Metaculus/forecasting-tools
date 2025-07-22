import logging

import pytest

from forecasting_tools.agents_and_tools.base_rates.deduplicator import Deduplicator

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "input_list, output_list",
    [
        (
            [
                "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
                "**1994 Moldovan referendum**: A referendum on remaining an independent nation was held on 6 March 1994 and approved by 97.9% of voters.",
            ],
            [
                "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
                "**1994 Moldovan referendum**: A referendum on remaining an independent nation was held on 6 March 1994 and approved by 97.9% of voters.",
            ],
        ),
        (
            [
                "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
                "**1994 Moldovan referendum**: A referendum on remaining an independent nation was held on 6 March 1994 and approved by 97.9% of voters.",
                "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
                "**1994 Moldovan referendum**: A referendum on remaining an independent nation was held on 6 March 1994 and approved by 97.9% of voters.",
            ],
            [
                "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
                "**1994 Moldovan referendum**: A referendum on remaining an independent nation was held on 6 March 1994 and approved by 97.9% of voters.",
            ],
        ),
        # (
        #     [
        #         "**H1N1 Influenza Pandemic**: A novel H1N1 influenza virus emerged in 2009, leading to a global pandemic affecting millions worldwide."
        #         "**West Africa Ebola Outbreak**: An Ebola virus epidemic that started in 2014 in West Africa, primarily affecting Guinea, Liberia, and Sierra Leone."
        #         "**Zika Virus Outbreak**: An outbreak of the Zika virus began in 2015, primarily affecting the Americas and linked to birth defects."
        #         "**COVID-19 Pandemic**: A global pandemic caused by the SARS-CoV-2 virus, starting in late 2019 and affecting nearly every country worldwide."
        #         "**H5N1 Avian Influenza**: Occurrences of H5N1 avian influenza in humans, posing a risk of potential pandemics in various years including 2013 and later."
        #         "**MERS Coronavirus Outbreak**: Middle East Respiratory Syndrome Coronavirus (MERS-CoV) outbreaks, primarily in the Middle East since 2012."
        #         "**Cholera Outbreak in Haiti**: Following the 2010 earthquake, a cholera outbreak occurred in Haiti, linked to international peacekeeping forces."
        #         "**Dengue Fever Outbreaks**: Dengue fever outbreaks in various regions like Southeast Asia and Latin America throughout the 2010s and 2020s."
        #         "**Yellow Fever Outbreak in Angola**: A large outbreak of yellow fever began in Angola in December 2015, spreading to other countries."
        #         "**Ebola Outbreak in Democratic Republic of the Congo**: Several Ebola outbreaks in the DRC, including a significant outbreak in 2018-2020."
        #         "**Monkeypox Outbreak**: A multi-country monkeypox outbreak reported across various regions in 2022."
        #         "**Polio Outbreaks**: Events of polio outbreaks in various regions, including resurgence in countries like Syria and the Philippines."
        #         "**H1N1 Influenza Pandemic**: The H1N1 pandemic, also known as swine flu, occurred in 2009. It was an influenza pandemic that spread across the world, resulting in significant public health concern."
        #         "**Ebola Outbreak in West Africa**: Between 2013 and 2016, West Africa experienced the largest Ebola outbreak in history, affecting countries like Guinea, Liberia, and Sierra Leone."
        #         "**Middle East Respiratory Syndrome (MERS)**: Since 2012, MERS outbreaks have been reported, particularly in the Arabian Peninsula, posing a risk to international health due to its high mortality rate."
        #         "**Zika Virus Outbreak**: In 2015-2016, the Zika virus outbreak in the Americas was declared a public health emergency due to its association with birth defects."
        #         "**COVID-19 Pandemic**: The COVID-19 pandemic, caused by the SARS-CoV-2 virus, began in Wuhan, China in late 2019 and resulted in widespread global health, economic, and social impacts."
        #         "**Monkeypox Outbreak**: In 2022, multiple countries reported monkeypox cases, prompting the WHO to declare it a public health emergency of international concern."
        #         "**Cholera Outbreak in Yemen**: Since 2016, Yemen has experienced a severe cholera outbreak, exacerbated by ongoing conflict, with large numbers of cases and deaths reported."
        #         "**Marburg Virus Outbreaks**: Marburg virus, a highly virulent pathogen similar to Ebola, has caused outbreaks in Africa, including recent events in 2023."
        #         "**Polio Resurgence**: Despite global eradication efforts, polio resurgence has been noted in certain regions, posing a threat to international public health."
        #         "**Drug-Resistant Tuberculosis**: The emergence of multidrug-resistant tuberculosis (MDR-TB) continues to pose significant challenges for global health systems."
        #         "**H5N1 Avian Influenza**: H5N1, also known as bird flu, has sporadically caused outbreaks in poultry and humans, raising concerns of a potential pandemic threat."
        #         "**H1N1 Influenza (Swine Flu) Pandemic**: The H1N1 influenza pandemic of 2009 was declared by WHO as a global pandemic as it spread to multiple countries worldwide."
        #         "**Cholera Outbreaks in Haiti**: Post the 2010 earthquake, Haiti experienced significant cholera outbreaks attributed to contaminated water, posing major international health concerns."
        #         "**Ebola Virus Epidemic in West Africa**: From 2013 to 2016, the Ebola outbreak in West Africa was unprecedented in scale and posed significant international health risk."
        #         "**MERS-CoV Outbreaks**: Middle East Respiratory Syndrome Coronavirus outbreaks since 2012, primarily in the Middle East, have had sporadic international cases, causing health concern."
        #         "**COVID-19 Pandemic**: The COVID-19 pandemic, caused by SARS-CoV-2, began in late 2019 and was declared a pandemic by WHO in March 2020, affecting nearly every country globally."
        #         "**Zika Virus Epidemic**: The Zika virus epidemic of 2015-2016 primarily affected South America and posed significant public health concerns, particularly for pregnant women."
        #         "**Polio Outbreak in the Philippines**: The World Health Organization confirmed a polio outbreak in the Philippines in 2019, posing a regional health risk."
        #         "**Ebola Outbreak in Democratic Republic of the Congo**: Several outbreaks occurred in the DRC, notably in 2018 and 2020, causing regional concern with a risk of international spread."
        #         "**Monkeypox Outbreak**: Beginning in 2022, monkeypox cases were reported globally, leading to concerns of international health risks."
        #         "**Ebola Virus Outbreak in West Africa**: In early 2014, cases of the Ebola virus were detected in Guinea and soon after in Liberia and Sierra Leone. It spread to seven other countries, causing more than 11,000 deaths."
        #         "**Polio Public Health Emergency of International Concern**: Declared a PHEIC in May 2014 due to a rise in polio cases in Africa and Asia. As of 2022, polio remains endemic in Afghanistan and Pakistan."
        #         "**COVID-19 Pandemic**: The COVID-19 pandemic brought the world to a near halt in 2020 and has killed nearly seven million people worldwide."
        #         "**Cholera Outbreak in Haiti**: Major cholera outbreak following the earthquake in 2010 affected more than a half million people."
        #         "**Cholera Outbreak in Yemen**: Ongoing cholera outbreak affecting more than a half million people."
        #         "**Cholera Outbreak in Zimbabwe**: An outbreak in Zimbabwe in 2008–09 killed more than four thousand people."
        #         "**Nipah Virus Outbreak in India**: In India, an outbreak of the deadly Nipah virus was limited to a single case due to a strong public health response."
        #         "**Imported Cholera Cases in Burkina Faso**: Two cases of cholera imported from Niger led to investigations and border surveillance, with no further cases reported."
        #         "**Rabies Outbreak in Rural Tanzania**: A multidisciplinary team contained a rabies outbreak by engaging with the affected community to identify and respond to the risk."
        #     ],
        #     [
        #         "**H1N1 Influenza Pandemic**: A novel H1N1 influenza virus emerged in 2009, leading to a global pandemic affecting millions worldwide."
        #         "**Zika Virus Outbreak**: An outbreak of the Zika virus began in 2015, primarily affecting the Americas and linked to birth defects."
        #         "**COVID-19 Pandemic**: A global pandemic caused by the SARS-CoV-2 virus, starting in late 2019 and affecting nearly every country worldwide."
        #         "**H5N1 Avian Influenza**: Occurrences of H5N1 avian influenza in humans, posing a risk of potential pandemics in various years including 2013 and later."
        #         "**MERS Coronavirus Outbreak**: Middle East Respiratory Syndrome Coronavirus (MERS-CoV) outbreaks, primarily in the Middle East since 2012."
        #         "**Dengue Fever Outbreaks**: Dengue fever outbreaks in various regions like Southeast Asia and Latin America throughout the 2010s and 2020s."
        #         "**Yellow Fever Outbreak in Angola**: A large outbreak of yellow fever began in Angola in December 2015, spreading to other countries."
        #         "**Polio Outbreaks**: Events of polio outbreaks in various regions, including resurgence in countries like Syria and the Philippines."
        #         "**Nipah Virus Outbreak in India**: In India, an outbreak of the deadly Nipah virus was limited to a single case due to a strong public health response."
        #         "**Rabies Outbreak in Rural Tanzania**: A multidisciplinary team contained a rabies outbreak by engaging with the affected community to identify and respond to the risk."
        #         "**Marburg Virus Outbreaks**: Marburg virus, a highly virulent pathogen similar to Ebola, has caused outbreaks in Africa, including recent events in 2023."
        #         "**Drug-Resistant Tuberculosis**: The emergence of multidrug-resistant tuberculosis (MDR-TB) continues to pose significant challenges for global health systems."
        #         "**West Africa Ebola Outbreak**: An Ebola virus epidemic that started in 2014 in West Africa, primarily affecting Guinea, Liberia, and Sierra Leone."
        #         "**Ebola Outbreak in Democratic Republic of the Congo**: Several Ebola outbreaks in the DRC, including a significant outbreak in 2018-2020."
        #         "**Cholera Outbreak in Haiti**: Following the 2010 earthquake, a cholera outbreak occurred in Haiti, linked to international peacekeeping forces."
        #         "**Cholera Outbreak in Yemen**: Since 2016, Yemen has experienced a severe cholera outbreak, exacerbated by ongoing conflict, with large numbers of cases and deaths reported."
        #         "**Cholera Outbreak in Zimbabwe**: An outbreak in Zimbabwe in 2008–09 killed more than four thousand people."
        #         "**Imported Cholera Cases in Burkina Faso**: Two cases of cholera imported from Niger led to investigations and border surveillance, with no further cases reported."
        #     ],
        # ),
    ],
)
async def test_deduplicate_in_batches(input_list: list[str], output_list: list[str]):
    observed_answer = await Deduplicator.deduplicate_list_in_batches(
        input_list,
        initial_semantic_threshold=0.85,
        prompt_context="I want a list of unique events",
    )
    stringified_observed_answer = "\n- ".join(observed_answer)
    logger.info(
        f"\nInput length: {len(input_list)}\nTarget length: {len(output_list)}\nObserved length: {len(observed_answer)}\nObserved answer:\n {stringified_observed_answer}"
    )
    assert len(observed_answer) == len(output_list)
    assert set(observed_answer) == set(output_list)


@pytest.mark.parametrize(
    "texts, additional_text, correct_answer",
    [
        (
            [
                "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
                "**1994 Moldovan referendum**: A referendum on remaining an independent nation was held on 6 March 1994 and approved by 97.9% of voters.",
            ],
            "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
            True,
        ),
        (
            [
                "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
                "**1994 Moldovan referendum**: A referendum on remaining an independent nation was held on 6 March 1994 and approved by 97.9% of voters.",
            ],
            "**2010 Moldovan Referendum**: The referendum on 5 September 2010 asked if the President of Moldova should be elected by popular vote. Of those who voted, 87.83% chose 'Yes', but the results were invalid due to low voter turnout.",
            True,
        ),
        (
            [
                "**2010 Moldovan Constitutional Referendum**: A referendum held to decide on whether to amend the constitution to change the method of electing the president. While it got a majority of votes in favor, it did not meet the required turnout threshold.",
            ],
            "**1994 Moldovan referendum**: A referendum on remaining an independent nation was held on 6 March 1994 and approved by 97.9% of voters.",
            False,
        ),
    ],
)
async def test_adding_item_to_list_rejects_or_accepts_properly(
    texts: list[str], additional_text: str, correct_answer: bool
):
    observed_answer = await Deduplicator.determine_if_item_is_duplicate(
        additional_text, texts
    )
    assert observed_answer == correct_answer
