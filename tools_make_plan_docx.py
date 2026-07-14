#!/usr/bin/env python3
"""Create simple Word documents from the RC-STAR planning Markdown files."""

from __future__ import annotations

import html
import re
import sys
import zipfile
from pathlib import Path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def clean_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def run(text: str, bold: bool = False) -> str:
    props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    space = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return f"<w:r>{props}<w:t{space}>{esc(text)}</w:t></w:r>"


def paragraph(text: str = "", style: str | None = None, bold: bool = False) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{run(clean_inline(text), bold=bold)}</w:p>"


def bullet(text: str) -> str:
    return paragraph(f"• {clean_inline(text)}", "ListParagraph")


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    cols = max(len(row) for row in rows)
    width = 9360 // max(cols, 1)
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for _ in range(cols))
    xml = [
        "<w:tbl>",
        "<w:tblPr>",
        '<w:tblW w:w="0" w:type="auto"/>',
        "<w:tblBorders>",
        '<w:top w:val="single" w:sz="6" w:space="0" w:color="BFBFBF"/>',
        '<w:left w:val="single" w:sz="6" w:space="0" w:color="BFBFBF"/>',
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="BFBFBF"/>',
        '<w:right w:val="single" w:sz="6" w:space="0" w:color="BFBFBF"/>',
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="D9D9D9"/>',
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="D9D9D9"/>',
        "</w:tblBorders>",
        "</w:tblPr>",
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]
    for r_index, row in enumerate(rows):
        xml.append("<w:tr>")
        for cell in row + [""] * (cols - len(row)):
            shade = '<w:shd w:fill="EAF2F8"/>' if r_index == 0 else ""
            xml.append(
                "<w:tc>"
                f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>'
                f"{paragraph(clean_inline(cell), bold=(r_index == 0))}"
                "</w:tc>"
            )
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        line = lines[i].strip()
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        is_separator = all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells)
        if not is_separator:
            rows.append(cells)
        i += 1
    return table(rows), i


def markdown_to_body(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    para: list[str] = []
    in_code = False

    def flush_para() -> None:
        if para:
            out.append(paragraph(" ".join(para)))
            para.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para()
            in_code = not in_code
            i += 1
            continue
        if in_code:
            out.append(paragraph(line, "Code"))
            i += 1
            continue
        if not stripped:
            flush_para()
            i += 1
            continue
        if stripped.startswith("|"):
            flush_para()
            tbl, i = parse_table(lines, i)
            out.append(tbl)
            continue
        if stripped.startswith("# "):
            flush_para()
            out.append(paragraph(stripped[2:], "Title"))
        elif stripped.startswith("## "):
            flush_para()
            out.append(paragraph(stripped[3:], "Heading1"))
        elif stripped.startswith("### "):
            flush_para()
            out.append(paragraph(stripped[4:], "Heading2"))
        elif stripped.startswith("- "):
            flush_para()
            out.append(bullet(stripped[2:]))
        elif re.match(r"^\d+\.\s+", stripped):
            flush_para()
            out.append(paragraph(stripped, "ListParagraph"))
        elif stripped.startswith(">"):
            flush_para()
            out.append(paragraph(stripped.lstrip("> ").strip(), "Quote"))
        else:
            para.append(stripped)
        i += 1

    flush_para()
    return "".join(out)


def document_xml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}"><w:body>'
        f"{body}"
        "<w:sectPr>"
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
        "</w:body></w:document>"
    )


def styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="180" w:after="100"/></w:pPr><w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:ind w:left="360"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:ind w:left="360" w:right="360"/><w:spacing w:before="120" w:after="120"/></w:pPr><w:rPr><w:i/><w:color w:val="595959"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="20"/></w:rPr></w:style>
</w:styles>'''


def write_docx(markdown_path: Path, docx_path: Path) -> None:
    body = markdown_to_body(markdown_path.read_text(encoding="utf-8"))
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>''',
        )
        zf.writestr(
            "_rels/.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>''',
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''',
        )
        zf.writestr("word/document.xml", document_xml(body))
        zf.writestr("word/styles.xml", styles_xml())


def main() -> int:
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("usage: tools_make_plan_docx.py source.md output.docx [source.md output.docx ...]", file=sys.stderr)
        return 2
    for i in range(1, len(sys.argv), 2):
        write_docx(Path(sys.argv[i]), Path(sys.argv[i + 1]))
        print(f"wrote {sys.argv[i + 1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
