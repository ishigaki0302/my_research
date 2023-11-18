from openai import OpenAI

class ChangePrompt:
    def __init__(self):
       self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key="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",
        ) 
    def send(self, prompt, subject, attribute):
        input_text = f"""
        ###指令###
        入力されたプロンプトを、答えがattributeになるような疑問文に書き換えなさい。
        また、主語はsubjectに固定しないさい。
    
        ###例###
        ##入力##
        prompt:The Space Needle is in downtown
        subject:The Space Needle
        attribute:Seattle
        ##出力##
        In which city's downtown is the Space Needle located ?
    
        ###入力###
        prompt:{prompt}
        subject:{subject}
        attribute:{attribute}
        ###出力###
        """
        chat_completion = self.client.chat.completions.create(
        messages=[
                {
                    "role": "system",
                    "content": input_text,
                }
            ],
            model="gpt-4",
        )
        return chat_completion.choices[0].message.content