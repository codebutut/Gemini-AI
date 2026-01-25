from google import genai
from google.genai import types

from gemini_agent.config.app_config import AppConfig
from gemini_agent.core import tools
from gemini_agent.utils.logger import get_logger

logger = get_logger(__name__)


class SubAgent:
    """
    A specialized agent that executes a specific objective.
    """

    # Default prompt engineering flags to be appended to every user prompt
    DEFAULT_FLAGS = "--gemini --prompt-engineering --clarity --precision --structure --measurable-outcomes --actionable-details --professional-terminology --concise-keywords"

    def __init__(self, name: str, model: str = None, api_key: str = None):
        self.name = name
        self.config = AppConfig()
        self.api_key = api_key or self.config.api_key
        self.model = model or self.config.model
        self.client = genai.Client(api_key=self.api_key)
        self.history: list[types.Content] = []

    async def run(self, objective: str, max_turns: int = 10) -> str:
        """
        Runs the sub-agent loop to achieve the objective.
        """
        logger.info(f"SubAgent '{self.name}' starting with objective: {objective}")

        system_instruction = (
            f"You are a specialized sub-agent named '{self.name}'.\n"
            f"Your objective is: {objective}\n"
            "Use the available tools to achieve this goal efficiently.\n"
            "When you have completed the task or gathered the necessary information, "
            "provide a comprehensive summary of your findings or actions."
        )

        # Initialize conversation
        self.history = []

        # Initial user message with prompt engineering flags
        prompt = objective
        # Check if flags are already present (even partially)
        if "--gemini" not in prompt:
            prompt = f"{prompt}\n\n{self.DEFAULT_FLAGS}"

        current_content = types.Content(
            role="user", parts=[types.Part.from_text(text=prompt)]
        )
        self.history.append(current_content)

        tools_config = tools.get_tool_config()

        # Generation config
        gen_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            tools=[tools_config],
            system_instruction=system_instruction,
        )

        turn = 0
        final_response = ""

        while turn < max_turns:
            turn += 1
            logger.info(f"SubAgent '{self.name}' Turn {turn}/{max_turns}")

            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model, contents=self.history, config=gen_config
                )

                if not response.candidates or not response.candidates[0].content:
                    return "Error: Empty response from model."

                candidate = response.candidates[0]
                self.history.append(candidate.content)

                # Check for tool calls
                parts = candidate.content.parts
                function_responses = []

                for part in parts:
                    if part.function_call:
                        fn_name = part.function_call.name
                        fn_args = part.function_call.args

                        # Execute tool
                        logger.info(f"SubAgent executing: {fn_name}")

                        try:
                            if fn_name in tools.TOOL_FUNCTIONS:
                                result = tools.TOOL_FUNCTIONS[fn_name](**fn_args)
                            else:
                                result = f"Error: Tool '{fn_name}' not found."
                        except Exception as e:
                            result = f"Error executing {fn_name}: {e}"

                        function_responses.append(
                            types.Part.from_function_response(
                                name=fn_name, response={"result": result}
                            )
                        )

                if function_responses:
                    # Send tool outputs back
                    self.history.append(
                        types.Content(role="user", parts=function_responses)
                    )
                else:
                    # No tool calls, this is the final answer or a question
                    try:
                        final_response = response.text
                    except Exception:
                        final_response = "[Response text unavailable]"
                    break

            except Exception as e:
                logger.error(f"SubAgent error: {e}")
                return f"SubAgent crashed: {e}"

        logger.info(f"SubAgent '{self.name}' finished.")
        return final_response
