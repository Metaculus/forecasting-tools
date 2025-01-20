import pytest
import requests

from forecasting_tools.util.misc import raise_for_status_with_additional_info


def test_raise_for_status_raises_error_properly() -> None:
    url_that_does_not_exist = "https://www.google.com/this_page_does_not_exist"
    response = requests.get(url_that_does_not_exist)
    with pytest.raises(requests.exceptions.HTTPError):
        raise_for_status_with_additional_info(response)


def test_raise_for_status_does_not_raise_error_improperly() -> None:
    url_that_exists = "https://www.google.com"
    response = requests.get(url_that_exists)
    raise_for_status_with_additional_info(response)
