import logging

import pytest

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.markdown_tree import MarkdownTree

logger = logging.getLogger(__name__)


class TestTextOfSectionAndSubsections:

    def test_text_of_section_and_subsections_single_section(self) -> None:
        tree = MarkdownTree(
            level=1,
            title="Test Section",
            section_content="# Test Section\nSome content",
            sub_sections=[],
        )

        expected_text = "# Test Section\nSome content"
        assert tree.text_of_section_and_subsections == expected_text

    def test_text_of_section_and_subsections_with_sub_sections(self) -> None:
        subsection1 = MarkdownTree(
            level=2,
            title="Subsection 1",
            section_content="## Subsection 1\nSub content 1",
            sub_sections=[],
        )

        subsection2 = MarkdownTree(
            level=2,
            title="Subsection 2",
            section_content="## Subsection 2\nSub content 2",
            sub_sections=[],
        )

        tree = MarkdownTree(
            level=1,
            title="Main Section",
            section_content="# Main Section\nMain content",
            sub_sections=[subsection1, subsection2],
        )

        expected_text = "# Main Section\nMain content\n## Subsection 1\nSub content 1\n## Subsection 2\nSub content 2"
        assert tree.text_of_section_and_subsections == expected_text

    def test_text_of_section_and_subsections_nested_sub_sections(self) -> None:
        nested_subsection = MarkdownTree(
            level=3,
            title="Nested Subsection",
            section_content="### Nested Subsection\nNested content",
            sub_sections=[],
        )

        subsection = MarkdownTree(
            level=2,
            title="Subsection",
            section_content="## Subsection\nSub content",
            sub_sections=[nested_subsection],
        )

        tree = MarkdownTree(
            level=1,
            title="Main Section",
            section_content="# Main Section\nMain content",
            sub_sections=[subsection],
        )

        expected_text = "# Main Section\nMain content\n## Subsection\nSub content\n### Nested Subsection\nNested content"
        assert tree.text_of_section_and_subsections == expected_text


class TestTurnMarkdownIntoReportSections:

    def test_turn_markdown_into_report_sections_empty_markdown(self) -> None:
        result = MarkdownTree.turn_markdown_into_report_sections("")
        assert result == []

    def test_turn_markdown_into_report_sections_single_header(self) -> None:
        markdown = "# Test Section\nSome content"
        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 1
        assert result[0].level == 1
        assert result[0].title == "Test Section"
        assert result[0].section_content == "# Test Section\nSome content"
        assert result[0].sub_sections == []

    def test_turn_markdown_into_report_sections_multiple_headers_same_level(
        self,
    ) -> None:
        markdown = """# Section 1
Content 1
# Section 2
Content 2"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 2
        assert result[0].title == "Section 1"
        assert result[0].section_content == "# Section 1\nContent 1"
        assert result[1].title == "Section 2"
        assert result[1].section_content == "# Section 2\nContent 2"

    def test_turn_markdown_into_report_sections_nested_headers(self) -> None:
        markdown = """# Main Section
Main content
## Subsection 1
Sub content 1
## Subsection 2
Sub content 2"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 1
        main_section = result[0]
        assert main_section.title == "Main Section"
        assert main_section.level == 1
        assert len(main_section.sub_sections) == 2

        assert main_section.sub_sections[0].title == "Subsection 1"
        assert main_section.sub_sections[0].level == 2
        assert (
            main_section.sub_sections[0].section_content
            == "## Subsection 1\nSub content 1"
        )

        assert main_section.sub_sections[1].title == "Subsection 2"
        assert main_section.sub_sections[1].level == 2
        assert (
            main_section.sub_sections[1].section_content
            == "## Subsection 2\nSub content 2"
        )

    def test_turn_markdown_into_report_sections_deep_nesting(self) -> None:
        markdown = """# Level 1
Content 1
## Level 2
Content 2
### Level 3
Content 3
#### Level 4
Content 4"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 1
        level1 = result[0]
        assert level1.title == "Level 1"
        assert len(level1.sub_sections) == 1

        level2 = level1.sub_sections[0]
        assert level2.title == "Level 2"
        assert len(level2.sub_sections) == 1

        level3 = level2.sub_sections[0]
        assert level3.title == "Level 3"
        assert len(level3.sub_sections) == 1

        level4 = level3.sub_sections[0]
        assert level4.title == "Level 4"

    def test_turn_markdown_into_report_sections_header_levels_mixed(
        self,
    ) -> None:
        markdown = """# Level 1
Content 1
## Level 2
Content 2
### Level 3
Content 3
## Level 2 Again
Content 2 again
# Level 1 Again
Content 1 again"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 2

        first_level1 = result[0]
        assert first_level1.title == "Level 1"
        assert len(first_level1.sub_sections) == 2

        assert first_level1.sub_sections[0].title == "Level 2"
        assert len(first_level1.sub_sections[0].sub_sections) == 1
        assert first_level1.sub_sections[0].sub_sections[0].title == "Level 3"

        assert first_level1.sub_sections[1].title == "Level 2 Again"

        second_level1 = result[1]
        assert second_level1.title == "Level 1 Again"

    def test_turn_markdown_into_report_sections_intro_content_without_header(
        self,
    ) -> None:
        markdown = """This is intro content without a header
# First Section
Section content"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 2
        assert result[0].level == 0
        assert result[0].title is None
        assert result[0].section_content == "This is intro content without a header"
        assert result[1].title == "First Section"

    def test_turn_markdown_into_report_sections_multiple_intro_lines(
        self,
    ) -> None:
        markdown = """First intro line
Second intro line
Third intro line
# First Section
Section content"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 2
        assert result[0].level == 0
        assert result[0].title is None
        expected_intro_content = "First intro line\nSecond intro line\nThird intro line"
        assert result[0].section_content == expected_intro_content

    def test_turn_markdown_into_report_sections_empty_first_section_kept(
        self,
    ) -> None:
        markdown = """# Empty Section

# Real Section
Real content"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 2
        assert result[0].title == "Empty Section"
        assert result[1].title == "Real Section"

    def test_turn_markdown_into_report_sections_header_with_extra_spaces(
        self,
    ) -> None:
        markdown = "#   Test Section   \nSome content"
        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 1
        assert result[0].title == "Test Section"

    def test_turn_markdown_into_report_sections_different_header_levels(
        self,
    ) -> None:
        markdown = """# H1
Content 1
## H2
Content 2
### H3
Content 3
#### H4
Content 4
##### H5
Content 5
###### H6
Content 6"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 1
        h1 = result[0]
        assert h1.level == 1
        assert h1.title == "H1"

        h2 = h1.sub_sections[0]
        assert h2.level == 2
        assert h2.title == "H2"

        h3 = h2.sub_sections[0]
        assert h3.level == 3
        assert h3.title == "H3"

        h4 = h3.sub_sections[0]
        assert h4.level == 4
        assert h4.title == "H4"

        h5 = h4.sub_sections[0]
        assert h5.level == 5
        assert h5.title == "H5"

        h6 = h5.sub_sections[0]
        assert h6.level == 6
        assert h6.title == "H6"

    def test_turn_markdown_into_report_sections_content_with_multiple_lines(
        self,
    ) -> None:
        markdown = """# Section
Line 1
Line 2
Line 3
## Subsection
Sub line 1
Sub line 2"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 1
        main_section = result[0]
        expected_main_content = "# Section\nLine 1\nLine 2\nLine 3"
        assert main_section.section_content == expected_main_content

        subsection = main_section.sub_sections[0]
        expected_sub_content = "## Subsection\nSub line 1\nSub line 2"
        assert subsection.section_content == expected_sub_content

    def test_turn_markdown_into_report_sections_only_intro_content(
        self,
    ) -> None:
        markdown = """This is the only content
No headers at all
Just plain text"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 1
        assert result[0].level == 0
        assert result[0].title is None
        expected_content = (
            "This is the only content\nNo headers at all\nJust plain text"
        )
        assert result[0].section_content == expected_content

    def test_turn_markdown_into_report_sections_empty_lines_in_content(
        self,
    ) -> None:
        markdown = """# Section
Line 1

Line 2

Line 3"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 1
        expected_content = "# Section\nLine 1\n\nLine 2\n\nLine 3"
        assert result[0].section_content == expected_content

    def test_turn_markdown_into_report_sections_complex_nesting_scenario(
        self,
    ) -> None:
        markdown = """# Main
Main content
## Sub A
Sub A content
### Sub Sub A
Sub Sub A content
## Sub B
Sub B content
### Sub Sub B1
Sub Sub B1 content
### Sub Sub B2
Sub Sub B2 content
# Another Main
Another main content"""

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 2

        first_main = result[0]
        assert first_main.title == "Main"
        assert len(first_main.sub_sections) == 2

        sub_a = first_main.sub_sections[0]
        assert sub_a.title == "Sub A"
        assert len(sub_a.sub_sections) == 1

        sub_sub_a = sub_a.sub_sections[0]
        assert sub_sub_a.title == "Sub Sub A"

        sub_b = first_main.sub_sections[1]
        assert sub_b.title == "Sub B"
        assert len(sub_b.sub_sections) == 2

        assert sub_b.sub_sections[0].title == "Sub Sub B1"
        assert sub_b.sub_sections[1].title == "Sub Sub B2"

        second_main = result[1]
        assert second_main.title == "Another Main"

    def test_heading_level_skipped_if_no_level_1_header(self) -> None:
        markdown = clean_indents(
            """
            ## Section 1
            Content 1
            #### Subsection 1
            Sub content 1
            # Section 2
            Content 2
            ### Subsection 2 with another hashtag #1
            Sub content 2
            """
        )

        result = MarkdownTree.turn_markdown_into_report_sections(markdown)

        assert len(result) == 2
        section_1 = result[0]
        assert section_1.title == "Section 1"
        assert section_1.level == 2
        assert section_1.section_content == "## Section 1\nContent 1"
        assert len(section_1.sub_sections) == 1
        subsection_1 = section_1.sub_sections[0]
        assert subsection_1.title == "Subsection 1"
        assert subsection_1.level == 4
        assert subsection_1.section_content == "#### Subsection 1\nSub content 1"
        section_2 = result[1]
        assert section_2.title == "Section 2"
        assert section_2.level == 1
        assert section_2.section_content == "# Section 2\nContent 2"
        assert len(section_2.sub_sections) == 1
        subsection_2 = section_2.sub_sections[0]
        assert subsection_2.title == "Subsection 2 with another hashtag #1"
        assert subsection_2.level == 3
        assert (
            subsection_2.section_content
            == "### Subsection 2 with another hashtag #1\nSub content 2"
        )


class TestReportSectionsToMarkdown:

    @pytest.mark.parametrize(
        "top_heading_level, expected_output",
        [
            (
                0,
                clean_indents(
                    """
                    Section 1
                    Content 1
                    # Subsection 1
                    - Sub content 1
                        - Sub content 1.1
                        - Sub content 1.2
                    Section 2
                    Content 2
                    # Subsection 2
                    Sub content 2
                    ## Subsection 2.1
                    Sub content 2.1
                    """
                ),
            ),
            (
                1,
                clean_indents(
                    """
                    # Section 1
                    Content 1
                    ## Subsection 1
                    - Sub content 1
                        - Sub content 1.1
                        - Sub content 1.2
                    # Section 2
                    Content 2
                    ## Subsection 2
                    Sub content 2
                    ### Subsection 2.1
                    Sub content 2.1
                    """
                ),
            ),
            (
                2,
                clean_indents(
                    """
                    ## Section 1
                    Content 1
                    ### Subsection 1
                    - Sub content 1
                        - Sub content 1.1
                        - Sub content 1.2
                    ## Section 2
                    Content 2
                    ### Subsection 2
                    Sub content 2
                    #### Subsection 2.1
                    Sub content 2.1
                    """
                ),
            ),
        ],
    )
    def test_report_section_changes_heading_level(
        self, top_heading_level: int, expected_output: str
    ) -> None:
        markdown = clean_indents(
            """
            # Section 1
            Content 1
            ## Subsection 1
            - Sub content 1
                - Sub content 1.1
                - Sub content 1.2
            # Section 2
            Content 2
            ## Subsection 2
            Sub content 2
            ### Subsection 2.1
            Sub content 2.1
            """
        )

        report_sections = MarkdownTree.turn_markdown_into_report_sections(markdown)
        markdown_text_modified = MarkdownTree.report_sections_to_markdown(
            report_sections, top_heading_level
        )
        logger.info(f"Original markdown: \n{markdown}")
        logger.info(f"markdown_text_modified: \n{markdown_text_modified}")
        logger.info(f"expected_output: \n{expected_output}")

        assert markdown_text_modified.strip() == expected_output.strip()

    @pytest.mark.parametrize(
        "top_heading_level, expected_output",
        [
            (
                0,
                "Section 1\nContent 1\n# Subsection 1\nSub content 1\nSection 2\nContent 2",
            ),
            (
                1,
                "# Section 1\nContent 1\n## Subsection 1\nSub content 1\n# Section 2\nContent 2",
            ),
            (
                2,
                "## Section 1\nContent 1\n### Subsection 1\nSub content 1\n## Section 2\nContent 2",
            ),
            (
                3,
                "### Section 1\nContent 1\n#### Subsection 1\nSub content 1\n### Section 2\nContent 2",
            ),
            (
                None,
                "## Section 1\nContent 1\n### Subsection 1\nSub content 1\n## Section 2\nContent 2",
            ),
        ],
    )
    def test_heading_level_changed_if_no_level_1_header(
        self, top_heading_level: int | None, expected_output: str
    ) -> None:
        markdown = (
            "## Section 1\nContent 1\n### Subsection 1\nSub content 1\n## Section 2\nContent 2"
            ""
        )

        report_sections = MarkdownTree.turn_markdown_into_report_sections(markdown)
        markdown_text = MarkdownTree.report_sections_to_markdown(
            report_sections, top_heading_level
        )
        logger.info(f"Original markdown: \n{markdown}")
        logger.info(f"markdown_text: \n{markdown_text}")
        logger.info(f"expected_output: \n{expected_output}")
        assert markdown_text.strip() == expected_output.strip()

    def test_errors_if_first_header_is_not_largest_level(self) -> None:
        markdown = """
## Section 1
Content 1
### Subsection 1
Sub content 1
# Section 2
Content 2
"""

        report_sections = MarkdownTree.turn_markdown_into_report_sections(markdown)
        assert len(report_sections) == 2
        with pytest.raises(ValueError):
            MarkdownTree.report_sections_to_markdown(report_sections, 2)

        MarkdownTree.report_sections_to_markdown(
            report_sections
        )  # does not error if not requested to change heading level

    def test_headerless_content_in_first_section_is_kept(self) -> None:
        markdown = "headerless content\n# Section 1\nContent 1"

        report_sections = MarkdownTree.turn_markdown_into_report_sections(markdown)
        markdown_text = MarkdownTree.report_sections_to_markdown(report_sections, 3)
        assert markdown_text.strip() == "headerless content\n### Section 1\nContent 1"
