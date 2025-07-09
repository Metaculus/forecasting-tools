from __future__ import annotations

import copy
import re

from pydantic import BaseModel, model_validator


class MarkdownTree(BaseModel):
    """
    Markdown text broken up into a tree of markdown sections where sections are divided by headers.
    """

    level: int  # Equal to number of hashtags in the header line (can be 0 if at beginning of document)
    title: str | None
    section_content: str
    sub_sections: list[MarkdownTree]
    # See validation functions at bottom of file

    @property
    def text_of_section_and_subsections(self) -> str:
        text = self.section_content
        for subsection in self.sub_sections:
            text += f"\n{subsection.text_of_section_and_subsections}"
        return text

    @classmethod
    def report_sections_to_markdown(
        cls,
        report_sections: list[MarkdownTree],
        top_heading_level: int | None = None,
    ) -> str:
        if top_heading_level is None:
            return "\n".join(
                [
                    section.text_of_section_and_subsections
                    for section in report_sections
                ]
            ).strip()

        if report_sections[0].level != min(
            section.level for section in report_sections
        ):
            raise ValueError(
                "First section must be at the highest heading level in order to change the heading levels recursively"
            )

        copied_report_sections = copy.deepcopy(report_sections)
        updated_sections = []
        for section in copied_report_sections:
            updated_sections.append(
                cls._update_headings_recursively(section, top_heading_level)
            )

        text = "\n".join(
            [
                section.text_of_section_and_subsections
                for section in updated_sections
            ]
        ).strip()
        return text

    @classmethod
    def _update_headings_recursively(
        cls,
        report_section: MarkdownTree,
        target_heading_level: int,
    ) -> MarkdownTree:
        text_of_section = report_section.section_content
        if text_of_section.startswith("#"):
            stripped_text = text_of_section.lstrip("#")
            text_with_new_heading = f"{'#' * target_heading_level} {stripped_text.lstrip()}".lstrip()
        else:
            text_with_new_heading = text_of_section

        report_section.section_content = text_with_new_heading
        for subsection in report_section.sub_sections:
            cls._update_headings_recursively(
                subsection, target_heading_level + 1
            )
        return report_section

    @classmethod
    def turn_markdown_into_report_sections(
        cls, markdown: str
    ) -> list[MarkdownTree]:
        final_heirarchial_sections: list[MarkdownTree] = []
        lines = markdown.splitlines()
        flattened_running_section_stack: list[MarkdownTree] = []

        for line in lines:
            line_is_header = re.match(r"^#{1,8} ", line)
            at_top_level = not flattened_running_section_stack
            should_create_new_header_section = line_is_header
            within_normal_section_at_non_header_line = (
                not at_top_level and not line_is_header
            )
            within_intro_section_without_header = (
                final_heirarchial_sections
                and not line_is_header
                and at_top_level
            )
            should_create_intro_section_without_header = (
                not final_heirarchial_sections
                and not line_is_header
                and at_top_level
            )

            if should_create_new_header_section:
                new_section = cls.__create_new_section_using_header_line(line)
                cls.__remove_sections_from_stack_until_at_level_higher_than_new_section_level(
                    flattened_running_section_stack, new_section.level
                )
                cls.__add_new_section_to_active_section_or_else_to_top_section_list(
                    flattened_running_section_stack,
                    final_heirarchial_sections,
                    new_section,
                )
                flattened_running_section_stack.append(new_section)
            elif within_normal_section_at_non_header_line:
                active_section = flattened_running_section_stack[-1]
                active_section.section_content += f"\n{line}"
            elif should_create_intro_section_without_header:
                final_heirarchial_sections.append(
                    MarkdownTree(
                        level=0,
                        title=None,
                        section_content=line,
                        sub_sections=[],
                    )
                )
            elif within_intro_section_without_header:
                assert (
                    len(final_heirarchial_sections) == 1
                    and final_heirarchial_sections[0].title is None
                )
                intro_section_without_header = final_heirarchial_sections[-1]
                intro_section_without_header.section_content += f"\n{line}"
            else:
                raise RuntimeError("Unexpected condition")
        final_heirarchial_sections = cls.__remove_first_section_if_empty(
            final_heirarchial_sections
        )
        return final_heirarchial_sections

    @staticmethod
    def __create_new_section_using_header_line(line: str) -> MarkdownTree:
        assert line.startswith("#")
        heading_level = len(line) - len(line.lstrip("#"))
        title = line.strip("# ").strip()
        section = MarkdownTree(
            level=heading_level,
            title=title,
            section_content=line,
            sub_sections=[],
        )
        return section

    @staticmethod
    def __remove_sections_from_stack_until_at_level_higher_than_new_section_level(
        section_stack: list[MarkdownTree], current_level: int
    ) -> None:
        while section_stack and section_stack[-1].level >= current_level:
            section_stack.pop()

    @staticmethod
    def __add_new_section_to_active_section_or_else_to_top_section_list(
        running_section_stack: list[MarkdownTree],
        final_sections: list[MarkdownTree],
        new_section: MarkdownTree,
    ) -> None:
        if running_section_stack:
            active_section = running_section_stack[-1]
            active_section.sub_sections.append(new_section)
        else:
            final_sections.append(new_section)

    @staticmethod
    def __remove_first_section_if_empty(
        sections: list[MarkdownTree],
    ) -> list[MarkdownTree]:
        if not sections:
            return []
        first_section = sections[0]
        if not first_section.section_content:
            return sections[1:]
        return sections

    @model_validator(mode="after")
    def validate_level(self: MarkdownTree) -> MarkdownTree:
        if self.level < 0:
            raise ValueError(f"Level {self.level} must be greater than 0")

        return self

    @model_validator(mode="after")
    def validate_subsection_levels(self: MarkdownTree) -> MarkdownTree:
        for subsection in self.sub_sections:
            if subsection.level <= self.level:
                raise ValueError(
                    f"Subsection level {subsection.level} must be greater than parent section level {self.level}"
                )
        return self

    @model_validator(mode="after")
    def validate_section_content_hashtags(self: MarkdownTree) -> MarkdownTree:
        if self.level > 0 and self.section_content:
            lines = self.section_content.splitlines()
            first_line = lines[0].strip()
            hashtag_count = len(first_line) - len(first_line.lstrip("#"))
            if hashtag_count != self.level:
                raise ValueError(
                    f"Section content starts with {hashtag_count} hashtags but header level is {self.level}"
                )
            for non_first_line in lines[1:]:
                if non_first_line.startswith("#"):
                    raise ValueError(
                        f"Section content contains a line that starts with a hashtag that is not the header line: {non_first_line}"
                    )
        return self
