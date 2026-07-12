from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import torch


class HuggingFaceChatModel:
    def __init__(
        self,
        model_name: str,
        use_4bit: bool,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ):

        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if use_4bit:

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )

        else:

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
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
                "content": self.system_prompt,
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
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
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