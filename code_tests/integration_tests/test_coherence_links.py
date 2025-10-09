from forecasting_tools import MetaculusClient


def test_coherence_links_api():
    client = MetaculusClient()
    new_id = client.post_question_link(
        question1_id=27353,
        question2_id=30849,
        direction=1,
        strength=2,
        link_type="causal",
    )

    links = client.get_links_for_question(question_id=27353)
    my_links = [link for link in links if link.id == new_id]
    assert len(my_links) == 1

    client.delete_question_link(new_id)
