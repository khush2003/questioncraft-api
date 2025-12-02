import json
import os
import random
from typing import Optional, Union
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from pdf2image import convert_from_bytes
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


from helper import convert_to_list, create_chunks_with_overlap, load_documents_from_file, save_documents_to_file
from enum import Enum

from pypdfium2 import PdfDocument

load_dotenv()

class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self, model_name: str, openai_api_key: Optional[str] = None, openai_api_base: str = "https://openrouter.ai/api/v1", **kwargs):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY_2')
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)
        
class Groq(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self, model_name: str, openai_api_key: Optional[str] = None, openai_api_base: str = "https://api.groq.com/openai/v1", **kwargs):
        openai_api_key = openai_api_key or os.getenv('GROQ_API_KEY')
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)
app = FastAPI()






embeddings = HuggingFaceInferenceAPIEmbeddings(
    model_name="ggrn/e5-small-v2",
    api_key= os.getenv('HF_KEY')
);

# vectorstore: FAISS = FAISS.from_texts(texts=["sa"], embedding=embeddings)
# vectorstore.save_local("vector_store")
# retreiver = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})


# new_vector_store = FAISS.load_local(
#     "vector_store", embeddings, allow_dangerous_deserialization=True
# )

# results = new_vector_store.similarity_search_with_score("Hello", 10)
# print(results)



llm = ChatOpenRouter(model_name="meta-llama/llama-3.2-3b-instruct:free")
llmLarge = ChatOpenRouter(model_name="nousresearch/hermes-3-llama-3.1-405b:free")
llmDefaultLarge = ChatOpenRouter(model_name="meta-llama/llama-3.1-405b-instruct:free")
llmQwen = ChatOpenRouter(model_name="qwen/qwen-2-7b-instruct:free")

llmSmall = ChatOpenRouter(model_name="meta-llama/llama-3.2-1b-instruct:free")
llm70b = ChatOpenRouter(model_name="meta-llama/llama-3.1-70b-instruct:free")
geminiPro = ChatOpenRouter(model_name="google/gemini-flash-1.5-8b-exp")
liquid40b = ChatOpenRouter(model_name="liquid/lfm-40b:free")
phi14b = ChatOpenRouter(model_name="microsoft/phi-3-medium-128k-instruct:free")

llmHERMES = ChatOpenRouter(model_name="nousresearch/hermes-3-llama-3.1-405b")
llmLAMA = ChatOpenRouter(model_name="meta-llama/llama-3.1-405b-instruct")


# system_prompt = "You answer in math"

# promptSummary = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("user", "Tell me a joke about {topic}.")
# ])

# openrouter_chain = promptSummary | groqLlama8b
# print(openrouter_chain.invoke({"topic": "banana"}))



# API
@app.get("/")
def read_root():
    return {"Hello": "World"}


class TextInput(BaseModel):
        text: str
        pageNum: int

class EmbeddingRequest(BaseModel):
    textInput: list[TextInput]
    chunkSize: Optional[int] = 180
    overlap: Optional[int] = 40


@app.post("/api/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    textInput = request.textInput
    chunkSize = request.chunkSize
    overlap = request.overlap

    if not textInput:
        raise HTTPException(status_code=400, detail="Text is required")

    allChunks = []
    for i in range(len(textInput)):
        chunks = create_chunks_with_overlap(textInput[i].text, chunkSize, overlap, textInput[i].pageNum)
        allChunks.extend(chunks)
    
    documents = []
    for chunk in allChunks:
        documents.append(Document(page_content=chunk['chunk'], metadata=chunk['metadata']))
        
    print(documents[-1])
    save_documents_to_file(documents)
            
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vectorstore: FAISS = FAISS.from_documents(documents=documents, ids=uuids, embedding=embeddings)
    vectorstore.save_local("vector_store")

    return {"chunks": allChunks}


# documents = load_documents_from_file()



# promptSummary = ChatPromptTemplate.from_messages(
#     [("system", "Write a summary of the main topics from the following:\\n\\n{context}")]
# )

# llm.temperature = 0
# # Instantiate chain
# chain = create_stuff_documents_chain(llm, promptSummary)

# # Invoke chain
# summary = chain.invoke({"context": documents})
# print(summary)

# promptTopics = ChatPromptTemplate.from_messages(
#     [("system", """Write a list of the {number} main topics and ensure that the number of topics is {number} only. You will be penalized if you write more than or less than {number} topics. Do not responed with other words, just the list of topics. Format of answer: [<Topic1>, <Topic2> ...]
#       Here is the context to write the topics from\\n\\n{context}""")]
# )

# chain = promptTopics | llm 

# result = chain.invoke({"context": summary, "number": 3})
# topics = convert_to_list(result.content)
# print(topics)







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







# parser = mcqParser
# prompt = mcqPrompt


# chain = prompt | llmLarge | parser

# topic = "Supply chain management"

# vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# results = vectorstore.similarity_search_with_relevance_scores(
#     topic,
#     k=6,
# )

# content = ""
# for res, score in results:
#     content += f"* {res.page_content} .Source of the chunk: [Pdf : {res.metadata}]\n\n"
#     # print(f"* {res.page_content} [{res.metadata}] -- [{score}]")

# print(content)
# print("------------------")
# print(topic)
# result = chain.invoke({"content": content, "topic": topic})
# print(result)


# def generate_question_and_answer(content):
#     """
#     Generates a multiple-choice question and the correct answer based on the provided content using LangChain.

#     :param content: Text content to base the question and answer on.
#     :return: Generated question, options, correct answer, and explanation.
#     """
#     # Define the response schema
#     response_schemas = [
#         ResponseSchema(name="question", description="The multiple-choice question"),
#         ResponseSchema(name="option_a", description="Option A for the question"),
#         ResponseSchema(name="option_b", description="Option B for the question"),
#         ResponseSchema(name="option_c", description="Option C for the question"),
#         ResponseSchema(name="option_d", description="Option D for the question"),
#         ResponseSchema(name="correct_answer", description="The correct answer for the question which should be one of the multiple-choice question"),
#     ]

#     # Create the output parser
#     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

#     # Define the prompt template
#     prompt = PromptTemplate(
#         input_variables=["content"],
#         template="""
#             Generate a multiple-choice question based on the following content.

#             Content: {content}

#             {format_instructions}
#         """,
#         partial_variables={"format_instructions": output_parser.get_format_instructions()},
#     ) 
    
#     # Create the LLMChain
#     chain = LLMChain(prompt=prompt, llm=llm)
    
#     # Run the chain with the content
#     result = chain.run(content)
    
#     # Parse the output using the JSON output parser
#     result = output_parser.parse(result)

#     question = result.get("question")
    
#     correct_answer = result.get("correct_answer")  # This should be one of 'A', 'B', 'C', 'D'
#     options = ["Option A", "Option B", "Option C", "Option D"]
    
#     answers = [result.get("option_a"),
#                result.get("option_b"),
#                result.get("option_c"),
#                result.get("option_d")
#               ]
    
#     pre_answer = ['A) ', 'B) ', 'C) ', 'D)']
#     options_answers = zip(options, answers)
#     correct_answer = [option for option, answer in options_answers if answer == correct_answer][0]

#     random.shuffle(options)
#     explanation = generate_explanation(question, correct_answer)
#     question = question + '\n' + " ".join([pre + " " + answer for  pre, answer  in zip(pre_answer, answers)])
#     return question, options, correct_answer, explanation

def generate_explanation(context):
    """
    Generates an explanation for the provided question and correct answer.

    :param question: The question for which to generate an explanation.
    :param correct_answer: The correct answer to the question.
    :return: Generated explanation as a string.
    """
    prompt = PromptTemplate(
        input_variables=["context"],
        template="Provide a detailed explanation for why the following question has the mentioned correct answer:\n\n{context}\n\nExplanation:"
    )
    chain = LLMChain(prompt=prompt, llm=llmQwen)
    return chain.run({"context": context})

# def generate_question(content):
#     """
#     Generates a multiple-choice question along with options and the correct answer based on the content.

#     :param content: Text content to generate a question from.
#     :return: Tuple containing the question, options, correct answer, and explanation.
#     """
#     question, options, correct_answer, explanation = generate_question_and_answer(content)
#     return question, options, correct_answer, explanation














@app.post("/query")
async def query(request: EmbeddingRequest):
    textInput = request.textInput
    chunkSize = request.chunkSize
    overlap = request.overlap

    if not textInput:
        raise HTTPException(status_code=400, detail="Text is required")

    allChunks = []
    for i in range(len(textInput)):
        chunks = create_chunks_with_overlap(textInput[i].text, chunkSize, overlap, textInput[i].pageNum)
        allChunks.extend(chunks)
    
    documents = []
    for chunk in allChunks:
        documents.append(Document(page_content=chunk['chunk'], metadata=chunk['metadata']))
        
    print(documents[-1])

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vectorstore: FAISS = FAISS.from_documents(documents=documents, ids=uuids, embedding=embeddings)
    vectorstore.save_local("vector_store")

    return {"chunks": allChunks}



@app.get("/api/documents")
async def get_documents():
    documents = load_documents_from_file()
    return {"documents": documents}



@app.get("/api/topics")
async def get_topics(number: int):
    documents = load_documents_from_file()
    print("started")

    promptSummary = ChatPromptTemplate.from_messages(
        [("system", "Write a summary of the main topics from the following:\\n\\n{context}")]
    )
    print("Passed 1")

    llm.temperature = 0
    chain = create_stuff_documents_chain(llm, promptSummary)
    print("Passed 2")
    print(documents)
    summary = chain.invoke({"context": documents})
    print("Passed 3")
    print(summary)

    promptTopics = ChatPromptTemplate.from_messages(
        [("system", """Write a list of the {number} main topics and ensure that the number of topics is {number} only. 
          You will be penalized if you write more than or less than {number} topics. Do not respond with other words, just the list of topics. 
          Format of answer: [<Topic1>, <Topic2> ...]
            Example Answer: [Topic1, Topic2, Topic3]
            Ensure that you only answer with list of topics.
            Here is the context to write the topics from\\n\\n{context}\n Topics: """)]
    )

    chain = promptTopics | llm 
    result = chain.invoke({"context": summary, "number": number}) 
    topics = convert_to_list(result.content)

    return {"topics": topics, "summary": summary}

#TODO: Failsafe when extracted text is empty, do not continue and tell the user that could not extract text from pdf

class QuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    PROBLEM_SOLVING = "problem_solving"
    ESSAY = "essay"
    

class QuestionRequest(BaseModel):
    topic: str
    types: list[QuestionType]


def getSummary(content):
    promptSummary = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of questions only, do not include answers, start with the questions directly, Questions:\\n\\n{context}")]
    )
    llm.temperature = 0.3
    chain = promptSummary | llmQwen 
    summary = chain.invoke({"context": content})
    return summary.content
@app.post("/api/questions")
async def get_questions(request: QuestionRequest):
    topic = request.topic
    types = request.types
    parsers = {
        QuestionType.MCQ: mcqParser,
        QuestionType.TRUE_FALSE: trueFalseParser,
        QuestionType.SHORT_ANSWER: shortAnswerParser,
        QuestionType.PROBLEM_SOLVING: problemSolvingParser,
        QuestionType.ESSAY: essayParser
    }

    prompts = {
        QuestionType.MCQ: mcqPrompt,
        QuestionType.TRUE_FALSE: trueFalsePrompt,
        QuestionType.SHORT_ANSWER: shortAnswerPrompt,
        QuestionType.PROBLEM_SOLVING: problemSolvingPrompt,
        QuestionType.ESSAY: essayPrompt
    }

    vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    search_results = vectorstore.similarity_search_with_relevance_scores(topic, k=4)

    content = ""
    for res, score in search_results:
        content += f"* {res.page_content} .Source of the chunk: [Pdf : {res.metadata}]\n\n"
    # print(content)
    

    async def result_generator():
        results = []
        first = True
        for qtype in types:
            parser = parsers[qtype]
            prompt = prompts[qtype]
            llmUse = llmLAMA
            if qtype == QuestionType.MCQ or qtype == QuestionType.TRUE_FALSE or qtype == QuestionType.SHORT_ANSWER or qtype == QuestionType.ESSAY:
                llmUse = llmHERMES
            
            if not first:
                summary = getSummary(results)
                print("Summary: ")
                print(summary)
            else:
                summary = ""
                first = False
                
            
            try:
                llmUse.temperature = 0
                chain = prompt | llmUse | parser
                result = chain.invoke({"content": content, "topic": topic, "previous_questions": summary})
                results.append({"type": qtype, "data": result})
                print("Result : ")
                print(result)
                yield json.dumps({"type": qtype, "data": result}) + "\n"
            except Exception as e:
                print(f"Error: {e}")
                try:
                    llmUse.temperature = 0.7
                    chain = prompt | llmUse | parser
                    result = chain.invoke({"content": content, "topic": topic, "previous_questions": summary})
                    results.append({"type": qtype, "data": result})
                    print("Result : ")
                    print(result)
                    yield json.dumps({"type": qtype, "data": result}) + "\n"
                except Exception as e:
                    print(f"Error: {e}")
                    try:
                        result = chain.invoke({"content": content, "topic": topic, "previous_questions": summary})
                        results.append({"type": qtype, "data": result})
                        print("Result : ")
                        print(result)
                        yield json.dumps({"type": qtype, "data": result}) + "\n"
                    except Exception as e:
                        print(f"Retry Error: {e}")
                        yield json.dumps({"type": "error", "data": {"error": e, "qtype": qtype}}) + "\n"
                        continue
                
    return StreamingResponse(result_generator(), media_type="application/json")


@app.post("/api/pdf")
async def pdf(file: UploadFile = File(...), fastMode: Optional[bool] = False):
    if file.content_type != "application/pdf":
        return {"error": "File must be a PDF"}

    # Read the file as bytes
    file_bytes = await file.read()
    from test import perform_ocr
    
    print(file)

    
    async def result_generator():
        texts = []
        pdf = PdfDocument(file_bytes)
        pages_count = len(pdf)
        yield json.dumps({"pages_count": pages_count}) + "\n"
        for i in range(pages_count):
            print("Processing page: ", i)
            page = pdf.get_page(i)
            if fastMode:
                text_page = page.get_textpage()
                text = text_page.get_text_range()
                abc = text
            else:
                image = page.render(scale=3)
                pil_image = image.to_pil()
                abc = perform_ocr(pil_image)
            texts.append(abc)
            yield json.dumps({"text": abc, "pageNum": i + 1}) + "\n"

    return StreamingResponse(result_generator(), media_type="application/json")


class ExplainationRequest(BaseModel):
    context: str

@app.post("/api/explaination")
async def get_explaination(request: ExplainationRequest):
    context = request.context

    explanation = generate_explanation(context)
    return {"explanation": explanation}


@app.get("/api/ask")
async def get_topics(input: str):
    documents = load_documents_from_file()
    

    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Answer the question in extensive detail as an academic expert with appropriate references. "
    "\n\n"
    "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )
    
    vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    retreiver = vectorstore.as_retriever(search_kwargs={"k": 9})
    llmLAMA.temperature = 0.4
    question_answer_chain = create_stuff_documents_chain(llmLAMA, prompt)
    rag_chain = create_retrieval_chain(retreiver, question_answer_chain)
    
    result = rag_chain.invoke({"input": input})
    return result['answer']
    
    # async def result_generator():
    #     async for chunk in rag_chain.astream({"input": input}):
    #         print(chunk)
    #         yield chunk
    
    
    # return StreamingResponse(result_generator(), media_type="application/json")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)