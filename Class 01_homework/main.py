# Homework Task:
#  Make a translation agent to translate text from one language to another.
# For your understandability, translate the code from urdu to english language.

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)



translator = Agent(
    name = 'Translator Agent',
    instructions = """You are a translator Agent. Translate the given text from Urdu to English.""",
)

response = Runner.run_sync(
    translator,
    input = "ادب و تہذیب کسی بھی قوم کی پہچان اور اس کے فکری ورثے کی عکاس ہوتی ہے۔ ایک مہذب معاشرے کی تعمیر میں اخلاقی اقدار، روایتی روایات اور فکری بصیرت اہم کردار ادا کرتے ہیں۔ ہر ذی شعور فرد پر لازم ہے کہ وہ اپنی زبان، ثقافت اور تہذیبی ورثے کی حفاظت کرے اور نئی نسل تک اسے احسن طریقے سے منتقل کرے۔ زبان صرف اظہارِ خیال کا ذریعہ نہیں بلکہ ایک مکمل تہذیبی شناخت ہے، جو قوموں کے عروج و زوال میں بنیادی حیثیت رکھتی ہے۔ اس لیے ضروری ہے کہ ہم اپنی علمی اور ادبی سرگرمیوں کو فروغ دیں اور قومی ہم آہنگی کے جذبے کو پروان چڑھاتے ہوئے ایک روشن اور مہذب معاشرے کی تشکیل میں اپنا مؤثر کردار ادا کریں۔",
    run_config = config
)

print(response.final_output)
