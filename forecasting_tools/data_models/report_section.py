from __future__ import annotations

import re

from pydantic import BaseModel


class ReportSection(BaseModel):
    level: int
    title: str | None
    section_content: str
    sub_sections: list[ReportSection]

    @property
    def text_of_section_and_subsections(self) -> str:
        text = self.section_content
        for subsection in self.sub_sections:
            text += f"\n{subsection.text_of_section_and_subsections}"
        return text

    @classmethod
    def turn_markdown_into_report_sections(
        cls, markdown: str
    ) -> list[ReportSection]:
        final_heirarchial_sections: list[ReportSection] = []
        lines = markdown.splitlines()
        flattened_running_section_stack: list[ReportSection] = []

        for line in lines:
            line_is_header = re.match(r"^#{1,6} ", line)
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
                    ReportSection(
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
    def __create_new_section_using_header_line(line: str) -> ReportSection:
        assert line.startswith("#")
        heading_level = line.count("#")
        title = line.strip("# ").strip()
        section = ReportSection(
            level=heading_level,
            title=title,
            section_content=line,
            sub_sections=[],
        )
        return section

    @staticmethod
    def __remove_sections_from_stack_until_at_level_higher_than_new_section_level(
        section_stack: list[ReportSection], current_level: int
    ) -> None:
        while section_stack and section_stack[-1].level >= current_level:
            section_stack.pop()

    @staticmethod
    def __add_new_section_to_active_section_or_else_to_top_section_list(
        running_section_stack: list[ReportSection],
        final_sections: list[ReportSection],
        new_section: ReportSection,
    ) -> None:
        if running_section_stack:
            active_section = running_section_stack[-1]
            active_section.sub_sections.append(new_section)
        else:
            final_sections.append(new_section)

    @staticmethod
    def __remove_first_section_if_empty(
        sections: list[ReportSection],
    ) -> list[ReportSection]:
        if not sections:
            return []
        first_section = sections[0]
        if not first_section.section_content:
            return sections[1:]
        return sections
