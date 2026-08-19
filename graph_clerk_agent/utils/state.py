from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage
import operator
from dataclasses import dataclass

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


@dataclass
class UserContext:
    number: str