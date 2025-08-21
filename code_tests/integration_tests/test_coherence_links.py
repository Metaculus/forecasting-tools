from forecasting_tools.helpers.metaculus_api import MetaculusApi


def test_coherence_links_api():
    new_id = MetaculusApi.post_question_link(
        question1_id=27353,
        question2_id=30849,
        direction="positive",
        strength="medium",
        link_type="causal",
    )

    links = MetaculusApi.get_links_for_question(question_id=27353)
    my_links = [link for link in links if link.id == new_id]
    assert len(my_links) == 1

    MetaculusApi.delete_question_link(new_id)
