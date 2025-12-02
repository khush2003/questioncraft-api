
from enum import Enum
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field




class MCQQuestion(BaseModel):
    question: str = Field(description="The multiple-choice question")
    option_a: str = Field(description="Option A for the question")
    option_b: str = Field(description="Option B for the question")
    option_c: str = Field(description="Option C for the question")
    option_d: str = Field(description="Option D for the question")
    correct_option: str = Field(description="The correct option (A,B,C or D) for the question")
    source: str = Field(description="The source of the question chunk in the form: Pdf : {'chunk_index': <number>, 'start_position': <number>, 'end_position': <number>, 'chunk_length': <number>, 'page_number': <number>}")

class MCQQuestions(BaseModel):
    questions: list[MCQQuestion] = Field(description="A list of multiple-choice questions")

mcqParser = JsonOutputParser(pydantic_object=MCQQuestions)

mcqPrompt = PromptTemplate(
    input_variables=["content", "previous_questions", "topic"],
    template="""Generate 10 multiple-choice questions which test the understanding of the topic {topic} from the following content:\\n\\n{content}\\n\\n{format_instructions} \n\n Please ensure that the questions are unique and not similar to the following questions: {previous_questions}""",
    partial_variables={"format_instructions": mcqParser.get_format_instructions()}
)

class TrueFalseQuestion(BaseModel):
    question: str = Field(description="The true or false question")
    correct_answer: bool = Field(description="The correct answer for the question")
    source: str = Field(description="The source of the question chunk in the form: Pdf : {'chunk_index': <number>, 'start_position': <number>, 'end_position': <number>, 'chunk_length': <number>, 'page_number': <number>}")

class TrueFalseQuestions(BaseModel):
    questions: list[TrueFalseQuestion] = Field(description="A list of true or false questions")

trueFalseParser = JsonOutputParser(pydantic_object=TrueFalseQuestions)

trueFalsePrompt = PromptTemplate(
    input_variables=["content", "previous_questions", "topic"],
    template="""Generate 10 true or false questions which test the understanding of the topic {topic} from the following content:\\n\\n{content}\\n\\n{format_instructions}\n\n Please ensure that the questions are unique and not similar to the following questions: {previous_questions}""",
    partial_variables={"format_instructions": trueFalseParser.get_format_instructions()}
)

class ShortAnswerQuestion(BaseModel):
    question: str = Field(description="The short answer question")
    correct_answer: str = Field(description="The correct answer for the question")
    source: str = Field(description="The source of the question chunk in the form: Pdf : {'chunk_index': <number>, 'start_position': <number>, 'end_position': <number>, 'chunk_length': <number>, 'page_number': <number>}")

class ShortAnswerQuestions(BaseModel):
    questions: list[ShortAnswerQuestion] = Field(description="A list of short answer questions")

shortAnswerParser = JsonOutputParser(pydantic_object=ShortAnswerQuestions)

shortAnswerPrompt = PromptTemplate(
    input_variables=["content", "previous_questions", "topic"],
    template="""Generate 10 short answer questions which test the understanding of the topic {topic} from the following content:\\n\\n{content}\\n\\n{format_instructions}\n\n Please ensure that the questions are unique and not similar to the following questions: {previous_questions}""",
    partial_variables={"format_instructions": shortAnswerParser.get_format_instructions()}
)

class EssayQuestion(BaseModel):
    question: str = Field(description="The essay question")

class EssayQuestions(BaseModel):
    questions: list[EssayQuestion] = Field(description="A list of essay questions")

essayParser = JsonOutputParser(pydantic_object=EssayQuestions)

essayPrompt = PromptTemplate(
    input_variables=["content", "previous_questions", "topic"],
    template="""Generate 5 open ended essay question which thouroughly tests the understanding of the topic {topic} from the following content:\\n\\n{content}\\n\\n{format_instructions}\n\n Please ensure that the questions are unique and not similar to the following questions: {previous_questions}""",
    partial_variables={"format_instructions": essayParser.get_format_instructions()}
)

class ProblemSolvingQuestion(BaseModel):
    question: str = Field(description="The problem-solving question")
    solution: str = Field(description="""The guide to answer the problem.""")

class ProblemSolvingQuestions(BaseModel):
    questions: list[ProblemSolvingQuestion] = Field(description="A list of problem-solving questions")

problemSolvingParser = JsonOutputParser(pydantic_object=ProblemSolvingQuestions)

problemSolvingPrompt = PromptTemplate(
    input_variables=["content" ,"previous_questions", "topic"],
    template="""Generate 5 problem-solving question and guides which test the understanding of the topic {topic},
    A good problem solving question guide:
    Evaluate the relevance and importance of elements that define the problem and limit the solutions.
    Identify information/data crucial to the problem and identify and prioritize patterns and trends in information/data most relevant to the problem.
    Generate a range of possible solutions that do not simply borrow from past examples and select and justify a chosen solution using evidence from an analysis of the options.
    Justify a data collection strategy by analyzing strengths and weaknesses and critiquing the potential effectiveness of a range of solutions with consideration of real-life constraints.
    Each answer must be answered in extensive details. With a step by step guide on how to solve the problem using the context.
    Use the following content:\\n\\n{content}\\n\\n{format_instructions}""",
    partial_variables={"format_instructions": problemSolvingParser.get_format_instructions()}
)

class QuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    PROBLEM_SOLVING = "problem_solving"
    ESSAY = "essay"
    