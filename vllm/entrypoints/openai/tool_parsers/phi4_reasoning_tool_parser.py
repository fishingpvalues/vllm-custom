# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from collections.abc import Sequence
from typing import Union

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


@ToolParserManager.register_module("phi4_reasoning")
class Phi4ReasoningToolParser(ToolParser):
    """
    Tool call parser for phi4-reasoning-plus models using Hermes-style extraction: finds all JSON objects with 'name' and 'arguments' fields (not a list, no tags).
    """

    TOOL_CALL_REGEX = re.compile(
        r'\{\s*"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*.*?\}', re.DOTALL
    )

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matches = self.TOOL_CALL_REGEX.findall(model_output)
        tool_calls = []
        for match in matches:
            try:
                call = json.loads(match)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=call["name"],
                            arguments=json.dumps(
                                call["arguments"], ensure_ascii=False
                            ),
                        ),
                    )
                )
            except Exception:
                continue
        content = (
            model_output.split(matches[0])[0].strip()
            if matches
            else model_output
        )
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=content if tool_calls else model_output,
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
        matches = self.TOOL_CALL_REGEX.findall(current_text)
        if not matches:
            return DeltaMessage(content=delta_text)
        try:
            call = json.loads(matches[-1])
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=len(matches) - 1,
                        type="function",
                        id="phi4_tool_call_stream",
                        function=DeltaFunctionCall(
                            name=call["name"],
                            arguments=json.dumps(
                                call["arguments"], ensure_ascii=False
                            ),
                        ).model_dump(exclude_none=True),
                    )
                ]
            )
        except Exception:
            return None
