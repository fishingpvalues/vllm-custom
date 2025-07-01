# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Union

import regex as re

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("deepseek_r1")
class DeepSeekR1ToolParser(ToolParser):
    """Tool parser for DeepSeek R1 models using XML-like tool call tags."""

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        # Tool call tags as per DeepSeek R1 chat template
        self.calls_begin = "<｜tool▁call▁begin｜>"
        self.calls_end = "<｜tool▁call▁end｜>"
        self.call_begin = "<｜tool▁call▁begin｜>"
        self.call_end = "<｜tool▁call▁end｜>"
        # Regex to extract tool call JSON blocks
        self.tool_call_block_re = re.compile(
            re.escape(self.call_begin) + r"(.*?)" + re.escape(self.call_end), re.DOTALL
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Only process if tool call tags are present
        if self.calls_begin not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        try:
            # Extract all tool call JSON blocks
            tool_calls = []
            for match in self.tool_call_block_re.findall(model_output):
                json_str = match.strip()
                if not json_str:
                    continue
                try:
                    call_obj = json.loads(json_str)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=call_obj["name"],
                                arguments=json.dumps(
                                    call_obj["parameters"], ensure_ascii=False
                                ),
                            ),
                        )
                    )
                except Exception:
                    logger.exception("Failed to parse tool call JSON: %s", json_str)
                    continue
            # Content is everything before the first tool call block
            content = model_output.split(self.calls_begin, 1)[0].strip()
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error extracting DeepSeek R1 tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        # Streaming: look for the last complete tool call block in the current text
        try:
            matches = list(self.tool_call_block_re.finditer(current_text))
            if not matches:
                return DeltaMessage(content=delta_text)
            last_match = matches[-1]
            json_str = last_match.group(1).strip()
            if not json_str:
                return None
            try:
                call_obj = json.loads(json_str)
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=0,  # Only one at a time in streaming
                            type="function",
                            id=random_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=call_obj["name"],
                                arguments=json.dumps(
                                    call_obj["parameters"], ensure_ascii=False
                                ),
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
            except Exception:
                logger.exception(
                    "Failed to parse streaming tool call JSON: %s", json_str
                )
                return None
        except Exception:
            logger.exception("Error extracting DeepSeek R1 streaming tool call.")
            return None
