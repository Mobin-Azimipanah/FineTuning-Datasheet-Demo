# FineTuning-Datasheet-Demo versus Retrieval Augmented Generation (RAG)

## Project Overview

Fine Tuning a QA model on the Datasheet using a pretrained model and Low-Rank Adaption (LoRA) and comparing the result obtanied using Retrieval Augmented Generation (RAG)


![Reference Image](/pic/1_F7uWJePoMc6Qc1O2WxmQqQ.png)
![Reference Image](/pic/360_F_565740155_jISgBIQ6KxAnJrU9BelT0monORTNarHm.jpg)


### Learning Objectives

- Preprocessing the Datasheet

- Apply LoRA to a model

- Fine-tune on your provided dataset

- Save your model

- Conduct inference using the fine-tuned model


You can learn more about Peft and LoRA here:
<https://huggingface.co/docs/peft/index>


## Data preparation
The project seems to focus on fine-tuning datasheets or technical documents related to electronic components. The goal appears to be extracting structured information from these documents using Python libraries like PyPDF2 for PDF handling and potentially Langchain for document extraction and metadata tagging.

List of Libraries used in Data preparation:

PyPDF2
OpenAI
LangChain


#### Usage
Extracting Information from PDF

The script uses PyPDF2 to extract text from a PDF file. It identifies details related to a specific electrical component. An example of the extracted information includes the component's technical specifications, product status links, and product summary.

Creating Metadata

Using Langchain's OpenAI functions, the script processes the extracted text to create structured metadata. This metadata is formatted based on a predefined schema, categorizing information into sections like specifications, product parameters, and product status links.

Generating Sample Sentences

Additionally, the code demonstrates how to generate sample sentences using the extracted metadata. It leverages the formatted metadata to create sentences describing the component's features and specifications.

Saving Extracted Information

Finally, the code includes a section to save the extracted information as a JSON file. This file contains a structured list of sentences summarizing the electrical component's details.



### Fine-Tuning


#### Code Overview:

1. Installation of Libraries:

The code begins by installing several Python packages. These include:

torch: PyTorch library.

peft==0.4.0: Package for efficient fine-tuning.

bitsandbytes==0.40.2: Library for quantization.

transformers==4.31.0: Hugging Face Transformers library.

trl==0.4.7: Temporal Reward Learning (TRL) library.

accelerate: A library to accelerate PyTorch training.


2. Spark Session Initialization and Dataset Loading:
A Spark session is initialized using SparkSession.builder.getOrCreate().
The code defines a class PartDataset and a static method get_data to load a JSON dataset using Spark.

3. Loading and Preprocessing Dataset:
The user is prompted to enter the input path for the dataset.

4. Filtering and Extracting Training Data:
The code filters out rows where the "message" column is not null.
The "message" column is then selected and printed.
The "message" key of all rows in the filtered dataset is extracted into a list called training_data.


5. Loading Pretrained Model and Tokenizer:
A base model and tokenizer are loaded from Hugging Face Transformers.
 

6. LoRA Config and Training Parameters:
LoRA (Long-Range) configuration and training parameters are defined.

7. Training the Model:
The model is fine-tuned using the specified parameters, and the results are saved.



### Recalling the trained Model

In the third file (fined_LLAMA) the fined tuned model merged with the pretrained model from huggingface and will be tested.

-----------------------------------------------------------

### In the Next Step we will utilize vector similarity techniques to enable the retrieval-augmented generator to extract relevant information from external sources.

you can consult the LangChain-RAG and Llama-Index code files. It's also worth noting that using the LangChain ChatBot in combination with the basic RAG model can lead to improved performance in question-answering tasks.




# Thanks For Following
