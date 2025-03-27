from forecasting_tools.util.file_manipulation import load_text_file


def test_readme_conversion_script_removed():
    readme = load_text_file("README.md")
    assert "pypistats" not in readme
    assert "jupyter nbconvert" not in readme

    assert "Discord" in readme
