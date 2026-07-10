from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


import torch


from config import *



class HuggingFaceChatModel:
    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            TRIPLET_MODEL_NAME,
            trust_remote_code=True
        )


        if TRIPLET_USE_4BIT:


            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,



                bnb_4bit_use_double_quant=True,
            )



            self.model = AutoModelForCausalLM.from_pretrained(

                TRIPLET_MODEL_NAME,

                quantization_config=quantization_config,
                device_map="auto",

                trust_remote_code=True,
            )



        else:

            self.model = AutoModelForCausalLM.from_pretrained(

                TRIPLET_MODEL_NAME,

                torch_dtype=torch.float16,

                device_map="auto",

                trust_remote_code=True,
            )





    def generate(
        self,
        user_prompt: str
    ) -> str:

        messages = [

            {
                "role": "system",

                "content": TRIPLET_SYSTEM_PROMPT,
            },


            {
                "role": "user",

                "content": user_prompt,
            },

        ]


        prompt = self.tokenizer.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,

        )

        inputs = self.tokenizer(

            prompt,

            return_tensors="pt",

        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,

            max_new_tokens=TRIPLET_MAX_NEW_TOKENS,
            temperature=TRIPLET_TEMPERATURE,
            top_p=TRIPLET_TOP_P,
            do_sample=TRIPLET_DO_SAMPLE,
            pad_token_id=self.tokenizer.eos_token_id,

        )


        generated_tokens = outputs[0][
            inputs.input_ids.shape[1]:

        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,

        )

        return response.strip()