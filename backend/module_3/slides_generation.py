from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

from backend.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL


class SlideContent(BaseModel):
    title: str = Field(min_length=1)
    subtitle: str = ""
    bullets: List[str] = Field(default_factory=list)
    layout_recommendation: Literal["title", "section", "content", "two_column", "quote"] = "content"
    picture_recommendation: str = Field(
        default="",
        description="A concrete, educational visual concept; empty when a visual would add no value.",
    )
    source_pages: str = Field(default="", description="Comma-separated source PDF page numbers")


class SlideDeck(BaseModel):
    title: str = Field(min_length=1)
    subtitle: str = ""
    theme_recommendation: str = Field(default="Modern academic, navy and blue with warm accents")
    slides: List[SlideContent] = Field(min_length=2)


SLIDE_DRAFTS: Dict[str, Dict[str, Any]] = {}
IMAGE_MODEL = "gpt-image-2"

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert instructional presentation designer. Create a grounded slide deck
using only the supplied book excerpts. Build a clear teaching narrative, not a chapter summary dump.

Requirements:
- Return exactly the requested number of slides, including a concise title slide.
- Use presentation-ready bullets (normally 5-7 per content slide).
- Spread coverage fairly across the selected sections and do not invent facts.
- Recommend one of the supported layouts for every slide.
- Give a specific picture/diagram recommendation when it improves understanding; never request
  decorative stock imagery or copyrighted characters.
- When image generation is requested, every slide must have a concrete picture_recommendation,
  including the title slide. Otherwise, picture_recommendation may be empty when no visual helps.
- Copy the relevant PDF page number(s) from chunk metadata into source_pages.
- Put no bullets on the title slide. Keep all text concise enough to fit a widescreen slide.
- Respect the audience, learning goal, and additional instructions.""",
    ),
    (
        "human",
        "Audience: {audience}\nLearning goal/instructions: {instructions}\n"
        "Image generation requested: {generate_images}\n"
        "Selected sections: {sections}\nSlide count: {slide_count}\n\nBook excerpts:\n{context}",
    ),
])

REVISION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Revise the supplied educational slide-deck draft using the user's feedback.
Preserve grounding, factual meaning, useful source page references, and the structured output shape.
Make all requested improvements that are supported by the original book excerpts. Do not mention
the revision process or the feedback in the deck. Keep slide copy concise and presentation-ready.
When image generation is requested, retain a concrete picture_recommendation on every slide.""",
    ),
    (
        "human",
        "Image generation requested: {generate_images}\nUser feedback:\n{feedback}\n\n"
        "Current draft:\n{draft}\n\nGrounding excerpts:\n{context}",
    ),
])


def _llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature)


def _image_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required to generate slide images.")
    return OpenAI(api_key=OPENAI_API_KEY, max_retries=2)


def _image_prompt(slide: Dict[str, Any]) -> str:
    bullets = "; ".join(str(item) for item in slide.get("bullets") or [])
    return (
        "Create a polished educational presentation visual in a clean modern editorial style. "
        "Landscape 3:2 composition, clear focal point, simple shapes, restrained navy, blue, teal, "
        "and warm amber palette, suitable for a classroom PowerPoint. "
        f"Visual direction: {slide['picture_recommendation']}. "
        f"Slide topic: {slide.get('title', '')}. "
        f"Supporting ideas: {slide.get('subtitle', '')}; {bullets}"
    )


def _generate_slide_images(deck: Dict[str, Any], previous_deck: Dict[str, Any] | None = None) -> None:
    """Attach cost-optimized image data to slides with a visual recommendation."""
    client: OpenAI | None = None
    previous_slides = (previous_deck or {}).get("slides") or []
    for index, slide in enumerate(deck.get("slides") or []):
        recommendation = str(slide.get("picture_recommendation") or "").strip()
        if not recommendation:
            recommendation = f"A clear educational concept illustration for {slide.get('title') or 'this topic'}"
            slide["picture_recommendation"] = recommendation
        previous = previous_slides[index] if index < len(previous_slides) else {}
        if (
            recommendation == str(previous.get("picture_recommendation") or "").strip()
            and previous.get("generated_image_base64")
        ):
            slide["generated_image_base64"] = previous["generated_image_base64"]
            continue
        try:
            client = client or _image_client()
            result = client.images.generate(
                model=IMAGE_MODEL,
                prompt=_image_prompt(slide),
                size="1536x1024",
                quality="low",
                output_format="jpeg",
                output_compression=70,
                n=1,
            )
            image_base64 = result.data[0].b64_json if result.data else None
            if not image_base64:
                raise ValueError("The image API returned no image data.")
            slide["generated_image_base64"] = image_base64
        except Exception as exc:
            slide["image_generation_error"] = f"Image unavailable ({type(exc).__name__})."


def _format_chunks(chunks: List[Dict[str, Any]], max_chars: int = 240_000) -> str:
    """Format a representative spread of chunks within a practical context window."""
    max_items = max(1, max_chars // 1_500)
    if len(chunks) > max_items:
        indexes = {
            round(index * (len(chunks) - 1) / (max_items - 1))
            for index in range(max_items)
        } if max_items > 1 else {0}
        selected_chunks = [chunks[index] for index in sorted(indexes)]
    else:
        selected_chunks = chunks
    item_budget = max(500, (max_chars - len(selected_chunks) * 10) // max(len(selected_chunks), 1))
    passages: List[str] = []
    used = 0
    for index, chunk in enumerate(selected_chunks, start=1):
        metadata = chunk.get("metadata") or {}
        passage = (
            f"Excerpt {index}\nHeading: {metadata.get('heading_path') or 'Unknown'}\n"
            f"PDF pages: {metadata.get('pages') or ', '.join(map(str, chunk.get('pages') or [])) or 'Unknown'}\n"
            f"Content:\n{chunk.get('content', '')}"
        )
        remaining = max_chars - used
        if remaining <= 0:
            break
        passages.append(passage[:min(remaining, item_budget)])
        used += len(passages[-1]) + 10
    return "\n\n---\n\n".join(passages)


def _public_draft(record: Dict[str, Any]) -> Dict[str, Any]:
    deck = record["deck"]
    public_slides = []
    for slide in deck["slides"]:
        public_slide = {
            key: value
            for key, value in slide.items()
            if key != "generated_image_base64"
        }
        public_slide["image_generated"] = bool(slide.get("generated_image_base64"))
        public_slides.append(public_slide)
    return {
        "draft_id": record["draft_id"],
        "book": record["book"],
        "sections": record["sections"],
        "delta": record["delta"],
        "audience": record["audience"],
        "instructions": record["instructions"],
        "generate_images": record["generate_images"],
        "images_generated": sum(bool(slide.get("generated_image_base64")) for slide in deck["slides"]),
        "image_generation_failures": sum(bool(slide.get("image_generation_error")) for slide in deck["slides"]),
        "revision": record["revision"],
        **{**deck, "slides": public_slides},
    }


def generate_slides(state: Dict[str, Any]) -> Dict[str, Any]:
    slide_count = min(max(int(state.get("slide_count", 10)), 3), 30)
    audience = str(state.get("audience") or "Students")
    instructions = str(state.get("instructions") or "Create a clear teaching presentation.")
    context = _format_chunks(state["chunks"])
    generator = _llm(0.25).with_structured_output(SlideDeck)
    deck = (GENERATION_PROMPT | generator).invoke({
        "audience": audience,
        "instructions": instructions,
        "generate_images": bool(state.get("generate_images", False)),
        "sections": ", ".join(section["title"] for section in state["sections"]),
        "slide_count": slide_count,
        "context": context,
    }).model_dump()
    if not deck.get("slides"):
        raise ValueError("The model did not generate any slides.")
    generate_images = bool(state.get("generate_images", False))
    if generate_images:
        _generate_slide_images(deck)

    draft_id = uuid.uuid4().hex
    record = {
        "draft_id": draft_id,
        "book": state["book"],
        "sections": state["sections"],
        "delta": state["delta"],
        "audience": audience,
        "instructions": instructions,
        "generate_images": generate_images,
        "chunks": state["chunks"],
        "deck": deck,
        "revision": 0,
    }
    SLIDE_DRAFTS[draft_id] = record
    return {"draft_id": draft_id, "result": _public_draft(record)}


def load_slide_draft(state: Dict[str, Any]) -> Dict[str, Any]:
    record = SLIDE_DRAFTS.get(state["draft_id"])
    if not record:
        raise ValueError("This slide draft expired. Generate it again.")
    return record


def revise_slides(state: Dict[str, Any]) -> Dict[str, Any]:
    feedback = str(state.get("feedback") or "").strip()
    if not feedback:
        raise ValueError("Enter feedback before requesting a revision.")
    generator = _llm(0.2).with_structured_output(SlideDeck)
    deck = (REVISION_PROMPT | generator).invoke({
        "feedback": feedback,
        "generate_images": bool(state.get("generate_images", False)),
        "draft": state["deck"],
        "context": _format_chunks(state["chunks"]),
    }).model_dump()
    if state.get("generate_images", False):
        _generate_slide_images(deck, previous_deck=state["deck"])
    record = {
        key: state[key]
        for key in (
            "draft_id", "book", "sections", "delta", "audience", "instructions",
            "generate_images", "chunks",
        )
    }
    record.update({"deck": deck, "revision": int(state.get("revision", 0)) + 1})
    SLIDE_DRAFTS[state["draft_id"]] = record
    return {"result": _public_draft(record)}


def safe_presentation_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "presentation"
