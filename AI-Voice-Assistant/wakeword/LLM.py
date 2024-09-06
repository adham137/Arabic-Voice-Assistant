
from langchain_core.prompts import ChatPromptTemplate



from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory



class llm:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.4, groq_api_key="", model_name="llama-3.1-70b-versatile")
        self.chat_history = ChatMessageHistory()
        
        self.examples = {
            'Question': 'ما هي عاصمة فرنسا؟ الجواب هو باريس.',
            'Search': 'ابحث عن آخر الأخبار حول الذكاء الاصطناعي. نتائج البحث مثال: "اختراقات الذكاء الاصطناعي في الرعاية الصحية"، "خوارزمية جديدة للذكاء الاصطناعي للتنبؤات المالية"، "تقدمات الذكاء الاصطناعي في السيارات ذاتية القيادة".',
            'New Calendar': 'إنشاء حدث جديد في التقويم: افتح تطبيق التقويم، أنشئ حدثًا جديدًا، عيّن العنوان إلى "اجتماع"، حدد التاريخ إلى الغد، احفظ الحدث.',
            'Read Calendar': 'أحداث التقويم لليوم: 10:00 صباحًا - اجتماع الفريق في غرفة المؤتمرات، 12:30 مساءً - غداء مع سارة في المقهى، 5:00 مساءً - موعد المشروع في المكتب.',
            'Send Emails': 'إرسال بريد إلكتروني: افتح تطبيق البريد الإلكتروني، اكتب بريدًا جديدًا، المرسل إليه "john.doe@example.com"، الموضوع "تذكير بالاجتماع"، النص "مرحبًا جون، تذكير بشأن اجتماعنا غدًا.", أرسل البريد الإلكتروني.',
            'Read Emails': 'أحدث بريد إلكتروني من أليس: الموضوع: "تحديث المشروع"، النص: "مرحبًا، أردت تحديثك بشأن حالة المشروع. كل شيء يسير على ما يرام ونتوقع أن نلتزم بالموعد النهائي.", التاريخ: "2024-08-29".',
            'Call contact': 'اتصل بمايك سميث: افتح تطبيق الهاتف، ابحث عن مايك سميث في جهات الاتصال، ابدأ المكالمة.',
            'new contact': 'إضافة جهة اتصال جديدة: افتح تطبيق جهات الاتصال، أضف جهة اتصال جديدة، الاسم "سارة جونسون"، الهاتف "555-1234"، احفظ جهة الاتصال.',
            'weather': 'الطقس في نيويورك ليوم 2024-08-30: درجة الحرارة 75°F، الحالة: غائم جزئيًا، الرطوبة: 60%.',
            'open app': 'فتح تطبيق سبوتيفاي: ابحث عن تطبيق سبوتيفاي على جهازك وفتحه.',
            'Read notification': 'أحدث الإشعارات: "رسالة جديدة من أليس: لديك رسالة جديدة.", "تذكير بالاجتماع: تذكير لاجتماع الفريق في الساعة 10:00 صباحًا."',
            'Translation': 'ترجم "Hello" إلى الفرنسية: Bonjour.',
            'Rejection': 'رفض طلب صداقة من سام: انتقل إلى قسم طلبات الصداقة، اختر الطلب من سام، أكد الرفض.',
            'Acceptance': 'قبول دعوة اجتماع من جولي: انتقل إلى قسم الدعوات، اختر الدعوة من جولي، أكد القبول.',
            'greetings': 'تحية المستخدم: رسالة مثال "مرحبًا! كيف يمكنني مساعدتك اليوم؟"',
            'alarm': 'ضبط المنبه: افتح تطبيق المنبه، أنشئ منبهًا جديدًا، عيّن الوقت إلى 7:00 صباحًا، احفظ المنبه.'
        }



    def chat(self, message, intent="Read Calendar"):

        context = self.examples[intent]

        prompt_get_answer = ChatPromptTemplate.from_messages([

            SystemMessagePromptTemplate.from_template("You are a personal assistant and your job is to answer the user's questions \
                        based on his intent and his intent is: ##{intent}##,\
                        if the user asks questions about a specific topic your answer should be like the below example delimited by ###\
                        Here is the steps on how to formulate your answer: \
                        1. Read the example and understand it carefully.\
                        2. Determine whether the user wants a piece of information or for you to do an action.\
                        3. If the user wants a piece of information provide it to him based on your example understanding, else if he wants an action to be done, follow the steps of the action in the ex .\
                        4. Always reply in Arabic , under no circumstances should your answer be in english or any other language other than Arabic.\
                                        \\n\\n###{context}###"), 

            MessagesPlaceholder(variable_name="chat_history"),

            HumanMessagePromptTemplate.from_template("{question}")


        ])

        chain2 = LLMChain(
            llm=self.llm,
            prompt=prompt_get_answer,
        )
        try:
            response = chain2.invoke({
            "intent": intent,
            "context": context,
            "chat_history": self.chat_history.messages, 
            "question": message
            })
            self.chat_history.add_user_message(message)
            self.chat_history.add_ai_message(response['text'])
            return response['text']
        except:
            return "اسف مسمعتش  قل مرة كمان"


