import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import torch
import lancedb
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import warnings
import time
# import pyttsx3
warnings.filterwarnings("ignore")

# engine = pyttsx3.init()
# engine.setProperty('rate', 150)     # Speed
# engine.setProperty('volume', 0.75)   # Volume (0.0 to 1.0)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)  # Change 1 to another index to try different voices



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_last_word_if(text, word):
    words = text.strip().rstrip('.!?').split()
    if words and words[-1].lower() == word.lower():
        words.pop()
    return ' '.join(words)
def load_model():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('C:/Users/Jon/Documents/Career/Projects/DNDAgenticAI/dnd-stats/rag_model', device_map="auto", quantization_config=quantization_config)
    return model, tokenizer
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("C:/Users/Jon/Documents/Career/Projects/DNDAgenticAI/dnd-stats/data/lancedb")

    return db.open_table("baldurs_gate")


def get_context(query: str, table, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    actual = table.search(query).limit(num_results).to_list()
    contexts = []
    urls = []
    for dict in actual:
        contexts.append(dict['text'])
        urls.append(dict['metadata']['url'])
    return "\n\n".join(contexts), urls

def generate_words_fast(messages):
    #    Get chat template as text
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Now tokenize with full settings
    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        return_attention_mask=True
    ).to(device)

    # Make sure pad_token_id is set for decoder-only models (e.g., GPT-style)
    if res_model.config.pad_token_id is None:
        res_model.config.pad_token_id = tokenizer.eos_token_id
    outputs = res_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,             # ~2 sentences worth of tokens
        do_sample=True,                # enables sampling
        temperature=0.7,               # moderate creativity
        top_p=0.9,                     # nucleus sampling
        num_beams=1,                   # not using beam search
        dola_layers='high'            # helps with shorter, concise answers (optional)
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split on the assistant marker
    assistant_response = text.split("<|assistant|>")[-1].strip()

    return assistant_response



# Initialize database connection
table = init_db()

# Load model
res_model, tokenizer = load_model()
res_model.config.pad_token_id = tokenizer.eos_token_id

# Set up model: "base", "small", "medium", "large-v2"
model_size = "medium"
model1 = WhisperModel(model_size, compute_type="int8", device='cuda') # Use "int8" or "float32" if needed
model3 = WhisperModel(model_size, compute_type="int8", device='cuda') # Use "int8" or "float32" if needed

samplerate = 16000
blocksize = 4000
audio_queue1 = queue.Queue()
audio_queue3 = queue.Queue()

# Callback to collect audio blocks
def callback1(indata, frames, time, status):
    audio_queue1.put(indata.copy())
def callback3(indata, frames, time, status):
    audio_queue3.put(indata.copy())

# Start audio stream
stream1 = sd.InputStream(samplerate=samplerate, channels=1, callback=callback1, blocksize=blocksize)
stream3 = sd.InputStream(samplerate=samplerate, channels=1, callback=callback3, blocksize=blocksize)

stream1.start()


rec_int = False

start_word = 'oscar'
stop_word = 'cancel'
end_word = 'go'


print("Listening with Whisper... (Ctrl+C to stop)")
try:
    buffer1 = np.empty((0,), dtype=np.float32)
    buffer2 = np.empty((0,), dtype=np.float32)
    buffer3 = np.empty((0,), dtype=np.float32)
    while True:
        block1 = audio_queue1.get()
        block1 = block1.flatten()
        buffer1 = np.concatenate((buffer1, block1))

    
        # 👇 if recording is active, collect from stream3
        if rec_int:
            while not audio_queue3.empty():
                block3 = audio_queue3.get()
                block3 = block3.flatten()
                buffer3 = np.concatenate((buffer3, block3))


        seconds = 2
        # Run recognition every ~5 seconds
        if len(buffer1) >= samplerate * seconds:
            segment1 = buffer1[:samplerate * seconds]
            buffer1 = buffer1[samplerate * seconds:]

            segments1, _ = model1.transcribe(segment1, language="en")
            segments1 = list(segments1)
            if segments1:
                txt = segments1[-1].text.strip().lower()

                if not rec_int and start_word in txt:
                    stream3.start()
                    buffer3 = np.empty((0,), dtype=np.float32)
                    rec_int = True
                    print('Oscar Recognized\n')

                elif rec_int and stop_word in txt:
                    print('Cancelled')
                    rec_int = False
                    stream3.stop()

                elif rec_int and end_word in txt:
                    print('\nGO Recognized')
                    
                    # Continue collecting any remaining blocks before stopping
                    while not audio_queue3.empty():
                        block3 = audio_queue3.get()
                        block3 = block3.flatten()
                        buffer3 = np.concatenate((buffer3, block3))

                    stream3.stop()

                    segment3 = buffer3
                    segments3, _ = model3.transcribe(segment3, language="en")
                    segments3 = list(segments3)
                    if segments3:
                        prompt = []
                        for seg in segments3:
                            prompt.append(seg.text.strip())
                        prompt = ' '.join(prompt)   
                        start_time = time.time()
                        prompt = remove_last_word_if(prompt, "go")
                        print(prompt)
                        # print(txt_record[txt_start+6:txt_end])
                        context, urls = get_context(prompt, table)
                        for url in urls[:3]:
                            print(url)
                        system_prompt = f"""You are a helpful Baldurs Gate guide named {start_word}, giving short two sentence answers to questions. 
                        Use the context below to help give an answer to the user that is only about the game Baldurs Gate:
                        {context}
                        """
                        messages = [
                            {"role": "system", "content": system_prompt,},
                            {"role": "user", "content": prompt},
                        ]

                        response = generate_words_fast(messages)
                        print(response)
                        rec_int = False
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Generated in {elapsed_time:.4f} seconds")

                        # engine.say(response)
                    




except KeyboardInterrupt:
    print("\nExiting...")
    stream1.stop()
