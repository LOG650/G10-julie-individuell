local section_started = false
local section_counters = {0, 0, 0, 0, 0, 0}
local current_chapter = 0
local equation_counter = 0
local toc_entries = {}

local function trim(text)
  return (text:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function section_label(level)
  local parts = {}

  if level == 1 then
    return tostring(section_counters[1]) .. ".0"
  end

  for i = 1, level do
    table.insert(parts, tostring(section_counters[i]))
  end

  return table.concat(parts, ".")
end

local function prefixed_inlines(label, content)
  local prefixed = pandoc.Inlines({})
  prefixed:insert(pandoc.Str(label))
  prefixed:insert(pandoc.Space())

  for _, inline in ipairs(content) do
    prefixed:insert(inline)
  end

  return prefixed
end

local function is_display_equation(paragraph)
  return #paragraph.content == 1
    and paragraph.content[1].t == "Math"
    and paragraph.content[1].mathtype == "DisplayMath"
end

local function styled_paragraph(style, inlines)
  return pandoc.Div(
    {pandoc.Para(inlines)},
    pandoc.Attr("", {}, {{"custom-style", style}})
  )
end

local function build_toc_blocks()
  local blocks = {
    styled_paragraph("TOC Heading", pandoc.Inlines({pandoc.Str("Innhold")})),
  }

  for _, entry in ipairs(toc_entries) do
    if entry.level <= 3 then
      local link = pandoc.Link(entry.text, "#" .. entry.identifier)
      table.insert(blocks, styled_paragraph("TOC " .. tostring(entry.level), pandoc.Inlines({link})))
    end
  end

  return blocks
end

local function format_document(doc)
  doc = doc:walk({
    Header = function(el)
      if not section_started then
        if pandoc.utils.stringify(el.content) ~= "Innledning" then
          return el
        end

        section_started = true
      end

      section_counters[el.level] = section_counters[el.level] + 1

      for i = el.level + 1, #section_counters do
        section_counters[i] = 0
      end

      if el.level == 1 then
        current_chapter = section_counters[1]
        equation_counter = 0
      end

      local label = section_label(el.level)
      el.content = prefixed_inlines(label, el.content)

      table.insert(toc_entries, {
        level = el.level,
        text = pandoc.Inlines(el.content),
        identifier = el.identifier,
      })

      return el
    end,

    Para = function(el)
      if not section_started or not is_display_equation(el) then
        return el
      end

      equation_counter = equation_counter + 1

      local equation_number = string.format("(%d.%d)", current_chapter, equation_counter)
      local equation_tex = trim(el.content[1].text)
      local equation_table = table.concat({
        "|  |  |",
        "| --- | ---: |",
        "| $$" .. equation_tex .. "$$ | " .. equation_number .. " |",
      }, "\n")

      return pandoc.read(equation_table, "markdown").blocks[1]
    end,
  })

  doc = doc:walk({
    Div = function(el)
      if el.identifier == "toc-placeholder" then
        return build_toc_blocks()
      end

      return el
    end,
  })

  return doc
end

function Pandoc(doc)
  return format_document(doc)
end
