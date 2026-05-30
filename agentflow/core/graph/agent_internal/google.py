"""Google GenAI request helpers for Agent."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from .constants import GOOGLE_THINKING_BUDGET_BY_EFFORT


logger = logging.getLogger("agentflow.agent")


class AgentGoogleMixin:
    """Google GenAI message conversion and request helpers."""

    def _convert_to_google_format(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list]:
        """Convert chat-completion style messages into Google GenAI content objects."""
        from google.genai import types

        system_instruction = None
        google_contents: list[types.Content] = []
        call_id_to_name: dict[str, str] = {}

        for message in messages:
            for tool_call in message.get("tool_calls", []) or []:
                tool_call_id = tool_call.get("id", "")
                function_name = tool_call.get("function", {}).get("name", "")
                if tool_call_id and function_name:
                    call_id_to_name[tool_call_id] = function_name

        pending_tool_parts: list = []

        def flush_tool_parts() -> None:
            # Gemini requires the number of function_response parts in a user
            # turn to equal the number of function_call parts in the preceding
            # model turn. Parallel tool calls produce one model Content with N
            # function_call parts, so all consecutive tool results must be
            # merged into a single user Content with N function_response parts.
            if pending_tool_parts:
                from google.genai import types

                google_contents.append(types.Content(role="user", parts=list(pending_tool_parts)))
                pending_tool_parts.clear()

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "tool":
                pending_tool_parts.append(
                    self._build_tool_response_part(message, content, call_id_to_name)
                )
                continue

            flush_tool_parts()

            if role == "system":
                system_instruction = self._handle_system_message(content, system_instruction)
            elif role == "assistant" and message.get("tool_calls"):
                google_contents.append(self._handle_assistant_with_tools(message))
            else:
                google_role = "model" if role == "assistant" else "user"
                google_contents.append(self._handle_regular_message(content, google_role))

        flush_tool_parts()

        return system_instruction, google_contents

    def _handle_system_message(self, content: Any, system_instruction: str | None) -> str | None:
        """Handle system role messages and accumulate system instruction."""
        if system_instruction is None:
            return str(content)
        return system_instruction + "\n" + str(content)

    def _handle_assistant_with_tools(self, message: dict[str, Any]) -> Any:
        """Handle assistant message with tool calls."""
        from google.genai import types

        parts: list[types.Part] = []
        content = message.get("content", "")
        text = str(content) if content else ""

        if text:
            parts.append(types.Part(text=text))

        first_fc = True
        for tool_call in message["tool_calls"]:
            part = self._create_function_call_part(tool_call, first_fc)
            if first_fc:
                self._set_thought_signature(part, tool_call)
                first_fc = False
            parts.append(part)

        return types.Content(role="model", parts=parts)

    def _create_function_call_part(self, tool_call: dict[str, Any], is_first: bool = False) -> Any:
        """Create a FunctionCall part from a tool call."""
        from google.genai import types

        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        try:
            function_args = json.loads(function.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            function_args = {}

        return types.Part(
            function_call=types.FunctionCall(
                name=function_name,
                args=function_args,
            )
        )

    def _set_thought_signature(self, part: Any, tool_call: dict[str, Any]) -> None:
        """Set thought signature on function call part for Gemini 2.5/3 thinking models."""
        # Gemini 2.5/3 thinking models require a thought_signature on
        # the first function call part when replaying history. Use the
        # real signature stored during response conversion when available;
        # fall back to the officially documented bypass value otherwise.
        sig_b64 = tool_call.get("thought_signature")
        if sig_b64:
            part.thought_signature = base64.b64decode(sig_b64)
        else:
            part.thought_signature = b"skip_thought_signature_validator"

    def _build_tool_response_part(
        self, message: dict[str, Any], content: Any, call_id_to_name: dict[str, str]
    ) -> Any:
        """Build a single FunctionResponse Part from a tool role message."""
        from google.genai import types

        tool_call_id = message.get("tool_call_id", "")
        function_name = call_id_to_name.get(
            tool_call_id,
            message.get("name", "") or tool_call_id or "unknown_function",
        )
        return types.Part.from_function_response(
            name=function_name,
            response={"result": str(content) if content else ""},
        )

    def _handle_regular_message(self, content: Any, role: str) -> Any:
        """Handle regular text or multimodal messages (assistant or user).

        When *content* is a list of OpenAI-style content parts (produced by
        ``_build_content`` in the converter), each part is translated to a
        ``google.genai.types.Part``.  Plain strings are handled as before.
        """
        from google.genai import types

        if isinstance(content, list):
            parts = self._content_parts_to_google(content)
            if not parts:
                parts = [types.Part(text="")]
            return types.Content(role=role, parts=parts)

        return types.Content(
            role=role,
            parts=[types.Part(text=str(content) if content else "")],
        )

    def _content_parts_to_google(self, parts: list[dict[str, Any]]) -> list[Any]:
        """Convert a list of OpenAI-style content parts to Google types.Part objects."""
        from google.genai import types

        google_parts: list[types.Part] = []
        converters = {
            "text": self._text_part_to_google,
            "image_url": self._image_part_to_google,
            "input_audio": self._audio_part_to_google,
            "document": self._document_part_to_google,
            "video": self._video_part_to_google,
        }
        for part in parts:
            converter = converters.get(part.get("type", ""))
            if converter:
                google_parts.extend(converter(part, types))

        return google_parts

    def _text_part_to_google(self, part: dict[str, Any], types: Any) -> list[Any]:
        text = part.get("text", "")
        if not text:
            return []
        return [types.Part(text=text)]

    def _image_part_to_google(self, part: dict[str, Any], types: Any) -> list[Any]:
        image_info = part.get("image_url", {})
        if not isinstance(image_info, dict):
            return []

        url = image_info.get("url", "")
        if not isinstance(url, str) or not url:
            return []

        if url.startswith("data:"):
            header, _, b64_data = url.partition(",")
            mime = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
            raw = base64.b64decode(b64_data)
            return [types.Part.from_bytes(data=raw, mime_type=mime)]

        if url.startswith("gs://"):
            return [types.Part.from_uri(file_uri=url, mime_type="image/jpeg")]

        # Google does not accept arbitrary external https:// URLs via Part.from_uri.
        # Fetch the image bytes and use Part.from_bytes instead.
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310  # nosec B310
                raw = resp.read()
                mime = resp.headers.get("Content-Type", "image/jpeg")
                return [types.Part.from_bytes(data=raw, mime_type=mime)]
        except Exception:
            logger.warning("Failed to fetch external image URL for Google: %s", url)
            return [types.Part(text="[Failed to load image]")]

    def _audio_part_to_google(self, part: dict[str, Any], types: Any) -> list[Any]:
        audio_info = part.get("input_audio", {})
        if not isinstance(audio_info, dict):
            return []

        b64_data = audio_info.get("data", "")
        fmt = audio_info.get("format", "wav")
        if not b64_data:
            return []

        raw = base64.b64decode(b64_data)
        return [types.Part.from_bytes(data=raw, mime_type=f"audio/{fmt}")]

    def _document_part_to_google(self, part: dict[str, Any], types: Any) -> list[Any]:
        doc_info = part.get("document", {})
        if not isinstance(doc_info, dict):
            return []

        if doc_info.get("text"):
            return [types.Part(text=doc_info["text"])]

        return self._binary_or_uri_parts_to_google(
            doc_info,
            types,
            default_mime="application/pdf",
        )

    def _video_part_to_google(self, part: dict[str, Any], types: Any) -> list[Any]:
        video_info = part.get("video", {})
        if not isinstance(video_info, dict):
            return []

        return self._binary_or_uri_parts_to_google(
            video_info,
            types,
            default_mime="video/mp4",
        )

    def _binary_or_uri_parts_to_google(
        self,
        media_info: dict[str, Any],
        types: Any,
        *,
        default_mime: str,
    ) -> list[Any]:
        mime = media_info.get("mime_type", default_mime)
        data = media_info.get("data")
        if data:
            raw = base64.b64decode(data)
            return [types.Part.from_bytes(data=raw, mime_type=mime)]

        url = media_info.get("url")
        if url:
            # Google does not accept arbitrary external https:// URLs via Part.from_uri.
            # Only gs:// URIs are safe to pass through.
            if url.startswith("gs://"):
                return [types.Part.from_uri(file_uri=url, mime_type=mime)]

            try:
                import urllib.request

                with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310  # nosec B310
                    raw = resp.read()
                    return [types.Part.from_bytes(data=raw, mime_type=mime)]
            except Exception:
                logger.warning("Failed to fetch external media URL for Google: %s", url)
                return [types.Part(text="[Failed to load media]")]

        return []

    def _convert_tools_to_google_format(self, tools: list) -> list:
        """Convert OpenAI-style tool definitions into Google FunctionDeclarations."""
        from google.genai import types

        google_tools = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                function = tool["function"]
                function_decl_kwargs = {
                    "name": function["name"],
                    "description": function.get("description", ""),
                }
                if "parameters" in function:
                    function_decl_kwargs["parameters_json_schema"] = function["parameters"]
                google_tools.append(types.FunctionDeclaration(**function_decl_kwargs))
        return google_tools

    def _build_google_config(
        self,
        system_instruction: str | None,
        tools: list | None,
        call_kwargs: dict[str, Any],
    ) -> Any:
        """Build a Google GenerateContentConfig instance."""
        from google.genai import types

        config_kwargs = {}
        structured_output = getattr(self, "output_schema", None) is not None
        text_like_output = self.output_type in ("text", "json")

        if "temperature" in call_kwargs:
            config_kwargs["temperature"] = call_kwargs.pop("temperature")
        if "max_tokens" in call_kwargs or "max_output_tokens" in call_kwargs:
            config_kwargs["max_output_tokens"] = call_kwargs.pop(
                "max_tokens",
                call_kwargs.pop("max_output_tokens", None),
            )

        cached_content = call_kwargs.pop("cached_content", None)
        if cached_content:
            # system_instruction is already inside the cache — don't resend it
            config_kwargs["cached_content"] = cached_content
        elif system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools and text_like_output and not structured_output:
            function_declarations = self._convert_tools_to_google_format(tools)
            if function_declarations:
                config_kwargs["tools"] = [types.Tool(function_declarations=function_declarations)]

        if (
            self.reasoning_config
            and isinstance(self.reasoning_config, dict)
            and text_like_output
            and not structured_output
        ):
            thinking_kwargs: dict[str, Any] = {"include_thoughts": True}
            budget = self.reasoning_config.get("thinking_budget")
            effort = self.reasoning_config.get("effort")
            thinking_level = self.reasoning_config.get("thinking_level")
            if thinking_level is not None:
                # Explicit thinking_level — preferred for Gemini 3 models.
                # Accepts the string form ("low"/"medium"/"high"/"minimal") or
                # the ThinkingLevel enum value directly.
                thinking_kwargs["thinking_level"] = thinking_level
            elif budget is not None:
                thinking_kwargs["thinking_budget"] = int(budget)
            elif effort and effort in GOOGLE_THINKING_BUDGET_BY_EFFORT:
                thinking_kwargs["thinking_budget"] = GOOGLE_THINKING_BUDGET_BY_EFFORT[effort]
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    async def _call_google_content_generation(
        self,
        google_contents: list,
        config: Any,
        stream: bool,
        structured_output: bool,
    ) -> Any:
        """Call Google content generation endpoints for text or structured-json output."""
        mode_suffix = " (json schema)" if structured_output else ""

        if stream:
            logger.debug(
                "Calling Google aio.models.generate_content_stream%s with model=%s",
                mode_suffix,
                self.model,
            )
            return await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=google_contents,
                config=config,
            )

        logger.debug(
            "Calling Google aio.models.generate_content%s with model=%s",
            mode_suffix,
            self.model,
        )
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=google_contents,
            config=config,
        )
        cached = (
            getattr(getattr(response, "usage_metadata", None), "cached_content_token_count", 0) or 0
        )
        if cached:
            logger.debug("Cache hit: %d cached tokens (Google)", cached)
        return response

    async def _call_google(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call Google GenAI text, image, video, or audio endpoints."""
        output_schema = getattr(self, "output_schema", None)
        structured_output = output_schema is not None

        if tools and structured_output:
            raise ValueError(
                "Google GenAI does not currently support combining tool calls (function calling) "
                "with structured JSON outputs when output_schema is provided. "
                "Please execute tools using plain text output, or separate your tool gathering "
                "and structured extraction nodes."
            )

        call_kwargs = {**self.llm_kwargs, **kwargs}

        # Peek before _build_google_config pops it, so we know whether a cache is active.
        has_explicit_cache = bool(call_kwargs.get("cached_content"))

        system_instruction, google_contents = self._convert_to_google_format(messages)
        config = self._build_google_config(system_instruction, tools, call_kwargs)

        # When an explicit cache is active, system_instruction is excluded from the
        # config (the static part lives inside the cache).  Any dynamic additions
        # — memory injections, skill prompts, per-request state — are preserved by
        # prepending them as a leading user message so the model still sees them.
        if has_explicit_cache and system_instruction:
            from google.genai import types

            google_contents = [
                types.Content(role="user", parts=[types.Part(text=system_instruction)]),
                *google_contents,
            ]

        if structured_output:
            if config is None:
                from google.genai import types

                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=output_schema,
                )
            else:
                config.response_mime_type = "application/json"
                config.response_schema = output_schema
            return await self._call_google_content_generation(
                google_contents,
                config,
                stream,
                structured_output=True,
            )

        if self.output_type in ("text", "json"):
            return await self._call_google_content_generation(
                google_contents,
                config,
                stream,
                structured_output=False,
            )

        if self.output_type == "image":
            prompt = self._extract_prompt(messages)
            logger.debug("Calling Google aio.models.generate_images with model=%s", self.model)
            return await self.client.aio.models.generate_images(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        if self.output_type == "video":
            prompt = self._extract_prompt(messages)
            logger.debug("Calling Google aio.models.generate_videos with model=%s", self.model)
            return await self.client.aio.models.generate_videos(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        if self.output_type == "audio":
            prompt = self._extract_prompt(messages)
            logger.debug("Calling Google aio.models.generate_audio with model=%s", self.model)
            return await self.client.aio.models.generate_audio(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        raise ValueError(f"Unsupported output_type '{self.output_type}' for Google provider")
