from openai import OpenAI

class ChangePrompt:
    def __init__(self):
       self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key="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",
        )
    def translate_jp(self, prompt, subject, attribute):
        input_text = f"""
        ###指令###
        入力された英語を日本語に翻訳しなさい。

        ###ルール###
        promptの主語はsubjectに固定しないさい。
        promptに翻訳したsubjectと全く同じつづりの言葉を必ず含みなさい。
    
        ###例###
        ##入力##
        subject:The Space Needle
        attribute:Seattle
        prompt:In which city's downtown is the Space Needle located ?
        ##出力##
        subject:スペース・ニードル
        attribute:シアトル
        prompt:スペース・ニードルはどの都市のダウンタウンにありますか？
    
        ###入力###
        subject:{subject}
        attribute:{attribute}
        prompt:{prompt}
        ###出力###
        """
        count = 0
        while True:
            chat_completion = self.client.chat.completions.create(
            messages=[
                    {
                        "role": "system",
                        "content": input_text,
                    }
                ],
                model="gpt-4",
            )
            new_prompt = chat_completion.choices[0].message.content
            if count >= 5:
                return {"subject":"", "attribute":"", "prompt":""}
            try:
                pairs = new_prompt.split("\n")
                dictionary = {pair.split(':')[0]: pair.split(':')[1] for pair in pairs}
                if dictionary["subject"] in dictionary["prompt"]:
                    return dictionary
                else:
                    count += 1
            except:
                count += 1
                continue

    def get_subject(self, prompt):
        input_text = f"""
        ###指令###
        入力されたプロンプトの主語を抽出しなさい。

        ###ルール###
        文中に主語と全く同じつづりの言葉で抽出しなさい。
    
        ###例###
        ##入力##
        In which city's downtown is the Space Needle located ?
        ##出力##
        the Space Needle
    
        ###入力###
        {prompt}
        ###出力###
        """
        while True:
            chat_completion = self.client.chat.completions.create(
            messages=[
                    {
                        "role": "system",
                        "content": input_text,
                    }
                ],
                model="gpt-4",
            )
            subject = chat_completion.choices[0].message.content
            if subject in prompt:
                return subject

    def send(self, prompt, subject, attribute):
        input_text = f"""
        ###指令###
        入力されたプロンプトを、答えがattributeになるような疑問文に書き換えなさい。

        ###ルール###
        主語はsubjectに固定しないさい。
        文中にsubjectと全く同じつづりの言葉を必ず含みなさい。
    
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
        while True:
            chat_completion = self.client.chat.completions.create(
            messages=[
                    {
                        "role": "system",
                        "content": input_text,
                    }
                ],
                model="gpt-4",
            )
            new_prompt = chat_completion.choices[0].message.content
            if subject in new_prompt:
                return new_prompt

def main():
    client = ChangePrompt()
    print(client.translate_jp("What is the citizenship of Giuseppe Angeli?", "Giuseppe Angeli", "Italy"))

main()