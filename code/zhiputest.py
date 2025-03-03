from dotenv import load_dotenv
import os
from openai import OpenAI 
load_dotenv()
api_keys = os.getenv("API_KEY")

generate_client = OpenAI(
    api_key=api_keys,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
) 

completion = generate_client.chat.completions.create(
    model="glm-4-flash",  
    messages=[    
        {"role": "system", "content": "你是一个专业的程序员，擅长编写OpenMP代码，并且能够根据用户的需求，提供相应的代码实现。"},    
        {"role": "user", "content": """请你作为专业的程序员，将这个for循环代码段和对应的openmp指令修改完整。只输出你修改的结果代码，不要过程。
         for循环代码段：
         for (j = 0; j <n; j++)
            {
                mean[j] = 0.0;  
                for (i = 0; i < n; i++)    
                    mean[j] += data[i][j];
                    mean[j] /= float_n;
            }
         openmp指令：
         #pragma omp parallel for
         """} 
    ],
    top_p=0.7,
    temperature=0.9) 
 
print(completion.choices[0].message.content)