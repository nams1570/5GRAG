import sys
sys.path.append("..")
from settings import config
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json

response_template = '''You will be given a question, the gold truth answer to that question, and a chunk of text.
Respond with 'No' if you can answer the question correctly using just the chunk given.
Respond with 'Yes' if you need more context than just the chunk given to answer the question correctly.
question: {question} \n
gold truth answer: {answer}\n
chunk: {chunk} \n
Instructions:
1. Respond only with Yes or No. One word only.'''
def get_response(client,chunk, question, answer, seed):

    response_prompt = response_template.format(question=question,answer=answer,chunk=chunk)
    print(response_prompt)
    response = client.chat.completions.create(
        model=config["MODEL_NAME"],
        messages=[
            {
                'role': 'user',
                'content': response_prompt
            }
        ],
        seed=seed,
    )
    return response.choices[0].message.content

def process_item(client, chunk,question,answer, seed, max_retries=3, delay=3):
    """Encapsulates the logic for processing a single item."""
    for attempt in range(1, max_retries + 1):
        try:
            gpt_response = get_response(client,chunk,question,answer, seed)
            response = json.loads(gpt_response)
            return {
                'question':question,
                'ground_truth':answer,
                'is_good_question':response,
            }
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                print(f"Max retries reached, skipping this item.")
                return {
                    'question': question,
                    'ground_truth': answer,
                    'is_good_question': None,
                }

def filter_questions_via_llm(question_objs):
    pass

if __name__ == "__main__":
    client = OpenAI(
        api_key=config["API_KEY"],
        timeout=60,
    )

    output_path = "./yes.json"
    obj = {
        "question": "How is the timing and numbering of slots and OFDM symbols aligned within a subframe for a given subcarrier spacing configuration, and what constraints are placed on a half-duplex UE's transmission and reception timing relative to downlink and uplink symbols within and across cells?",
        "ground_truth": "Slots are numbered in increasing order within a subframe and frame for a given subcarrier spacing configuration. Each slot contains a certain number of consecutive OFDM symbols, where the number of symbols depends on the cyclic prefix length as specified in Tables 4.3.2-1 and 4.3.2-2. The start of each slot in a subframe is aligned with the start of the corresponding OFDM symbol in the same subframe, creating a fixed timing structure. OFDM symbols within a slot are classified as 'downlink', 'flexible', or 'uplink', which governs when a UE can transmit or receive. For a half-duplex UE (one not capable of full-duplex communication), transmission in uplink is not expected to occur earlier than a defined transition time after the end of the last received downlink symbol, and reception in downlink is not expected earlier than a defined transition time after the last uplink transmission, across all cells in a group. These transition times depend on parameters in Table 4.3.2-3, ensuring the UE respects timing advances and half-duplex restrictions. The detailed timing advance values and slot/OFDM numbering concepts come from the symbol definitions and timing parameters referenced in the first part, which relate to the structure and alignment of resource elements, symbols, and slots. Full timing and numbering understanding requires both the symbol definitions and slot timing plus half-duplex constraints sections.",
        "primary_chunk_section": "3.2",
        "primary_chunk_text": "For the purposes of the present document, the following symbols apply:\tResource element with frequency-domain index  and time-domain index  for antenna port  and subcarrier spacing configuration ; see clause 4.4.3\tValue of resource element  for antenna port and subcarrier spacing configuration ; see clause 4.4.3\tAmplitude scaling for a physical channel/signal\tPN sequence; see clause 5.2.1\tSubcarrier spacing\tSubcarrier spacing for random-access preambles\tThe ratio between  and ; see clause 4.1\tSubcarrier index relative to a reference\tOFDM symbol index relative to a reference\tSubcarrier spacing configuration, \tNumber of coded bits to transmit on a physical channel [for codeword ]\tNumber of modulation symbols to transmit on a physical channel [for codeword ]\tNumber of modulation symbols to transmit per layer for a physical channel\tScheduled bandwidth for uplink transmission, expressed as a number of subcarriers \tScheduled bandwidth for uplink transmission, expressed as a number of resource blocks\tNumber of modulation symbols to transmit per antenna port for a physical channel\tNumber of transmission layers\tSize of bandwidth part ; see clause 4.4.4.4\tStart of bandwidth part ; see clause 4.4.4.4\tCyclic prefix length; see clause 5.3.1\tThe size of the resource grid; see clauses 4.4.2 and 5.3\tThe start of the resource grid; see clause 4.4.2\tThe number of PT-RS groups; see clause 6.3.1.4\tPhysical layer cell identity; see clause 7.4.2.1\tPhysical-layer sidelink identity; see clause 8.4.2.1\tFrequency-domain size of a control resource set; see clause 7.3.2.2\tNumber of resource-element groups in a CORESET; see clause 7.3.2.2\tNumber of samples per PT-RS group; see clause 6.3.1.4\tNumber of subcarriers per resource block, see clause 4.4.4.1\tNumber of slots per subframe for subcarrier spacing configuration , see clause 4.3.2\tNumber of slots per frame for subcarrier spacing configuration , see clause 4.3.2\tTime duration of a control resource set; see clause 7.3.2.2\tLength of the PUCCH transmission in OFDM symbols; see clause 6.3.2.1\tNumber of OFDM symbols per subframe for subcarrier spacing configuration ; see clause 4.3.1\tNumber of symbols per slot\tTiming advance between downlink and uplink; see clause 4.3.1\tA fixed offset used to calculate the timing advance; see clause 4.3.1\tNetwork-controlled timing correction; see clause 4.3.1\tUE-derived timing correction; see clause 4.3.1\tMinimum time from reception to transmission for a half-duplex UE; see clause 4.3.2\tSystem frame number (SFN)\tCommon resource block number for subcarrier spacing configuration , see clause 4.4.4.3\tHyper-frame number\tPhysical resource block number; see clause 4.4.4.4\tRadio network temporary identifier\tSlot number within a subframe for subcarrier spacing configuration ; see clause 4.3.2\tSlot number within a frame for subcarrier spacing configuration ; see clause 4.3.2\tAntenna port number\tModulation order\tNumber of antenna ports\tLow-PAPR base sequence; see clause 5.2.2\tLow-PAPR sequence; see clause 5.2.2\tThe time-continuous signal on antenna port  and subcarrier spacing configuration  for OFDM symbol  in a subframe; see clause 5.3.1\tBasic time unit for NR; see clause 4.1\tRadio frame duration; see clause 4.3.1\tBasic time unit for LTE\tSubframe duration; see clause 4.3.1\tSlot duration; see clause 4.3.2\tTiming advance between downlink and uplink; see clause 4.3.1\tPrecoding matrix for spatial multiplexing"
    }
    print(process_item(client,obj["primary_chunk_text"],obj["question"],obj["ground_truth"],0))