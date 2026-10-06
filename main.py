"""Базовый агент на LangChain: промпт, свой tool, структурированный ответ и память."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Не найден OPENAI_API_KEY. Проверь файл .env")

SYSTEM_PROMPT = """Ты HR-помощник компании. Отвечай кратко и по делу, на русском языке.
Правило компании: ежегодный оплачиваемый отпуск составляет 28 календарных дней.
Если вопрос про количество дней отпуска за отработанные месяцы, обязательно
используй инструмент calculate_vacation_days и не считай в уме.
Если в вопросе нет числа отработанных месяцев, уточни его у пользователя.
Если пользователь спрашивает, о чём он говорил раньше, опирайся только на
сообщения этого диалога. Если в диалоге нет прошлых сообщений, прямо скажи,
что это начало диалога и ты ничего не помнишь. Ничего не выдумывай."""


@tool
def calculate_vacation_days(months_worked: int) -> str:
    """Считает, сколько дней отпуска накопилось за отработанные месяцы.

    Args:
        months_worked: количество полных отработанных месяцев (от 1 до 12).
    """
    if months_worked < 1 or months_worked > 12:
        return "Укажите число полных месяцев от 1 до 12."
    days = round(28 / 12 * months_worked, 1)
    return f"За {months_worked} мес. накоплено {days} календарных дней отпуска."


@dataclass
class ResponseFormat:
    """Формат ответа агента."""
    answer: str
    vacation_days: float | None = None


model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=API_KEY,
    base_url="https://api.openai.com/v1",
    temperature=0,
    max_tokens=1500,
)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[calculate_vacation_days],
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=InMemorySaver(),
)


def ask(question: str, thread_id: str) -> None:
    """Задаёт вопрос агенту в рамках диалога thread_id и печатает ответ."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    structured = result["structured_response"]
    print(f"\n[диалог {thread_id}] Вопрос: {question}")
    print(f"Ответ: {structured.answer}")
    print(f"Дней отпуска (структурно): {structured.vacation_days}")


if __name__ == "__main__":
    # Диалог 1: агент использует tool и помнит предыдущие сообщения
    ask("Сколько дней отпуска мне положено, если я отработала 6 месяцев?", "hr-1")
    ask("Что я только что спрашивала? Напомни вопрос и ответ одной фразой.", "hr-1")

    # Диалог 2: новый thread_id, памяти о первом диалоге нет
    ask("Что я только что спрашивала? Напомни вопрос и ответ одной фразой.", "hr-2")