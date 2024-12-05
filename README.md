
## Project Title: Arabic Smart Assistant

### Overview
The Arabic Smart Assistant is an innovative application designed to understand and respond to spoken Arabic using advanced machine learning models and natural language processing (NLP) techniques. The system integrates various technologies to provide a seamless conversational experience for Arabic-speaking users.

### Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [System Architecture](#system-architecture)
4. [Data and Model Training](#data-and-model-training)
5. [System Workflow](#system-workflow)
6. [Implementation Details](#implementation-details)
7. [Challenges and Solutions](#challenges-and-solutions)
8. [Future Work and Improvements](#future-work-and-improvements)
9. [Conclusion](#conclusion)

### Introduction
The project aims to create a smart assistant that can detect a specific trigger word, transcribe spoken Arabic into text, classify the intent of the transcribed text, and generate appropriate responses in Arabic. The system utilizes an LSTM model for trigger word detection, Google Speech-to-Text for transcription, an Arabic BERT model for intent classification, and a LLAMA model for response generation.

### Objectives
- Develop a robust system for detecting a trigger word using a custom-trained LSTM model.
- Implement accurate transcription of spoken Arabic using Google Speech-to-Text.
- Classify transcribed Arabic text into predefined intents using a fine-tuned Arabic BERT model.
- Generate coherent Arabic responses using a LLAMA model.
- Deliver responses in spoken Arabic via Google Text-to-Speech.

### System Architecture
The architecture consists of several key components:
- **Speech Input**: Captures spoken Arabic from the user.
- **Speech Recognition**: Converts speech to text using Google Speech-to-Text.
- **Trigger Word Detection**: Monitors input for predefined trigger words using an LSTM model.
- **Intent Classification**: Identifies user intent using a fine-tuned Arabic BERT model.
- **Response Generation**: Generates responses using a LLAMA model.
- **Output**: Converts generated text back to speech using Google Text-to-Speech.

### Data and Model Training
The project involves collecting and preprocessing data for both trigger word detection and intent classification.
1. The LSTM model is trained on a combination of datasets that are not included in this repositories.
  * The first dataset was created using scripts located in AI-Voice-Assistant/wakeword/scripts, which would include a number of my recordings saying the wake word split and duplicated.
  * The second dataset is using the "Common Voice" dataset, Using arabic and english recordings*
2. The Arabic BERT model is fine-tuned on a dataset of Arabic intent classfication found on kaggle.
  * The BERT model was fine-tunned on an [arabic intent classification dataset](https://drive.google.com/uc?id=1h620Wmx1yvkTKibH6N2wCNddOakBTUzg) present on google drive.
  * A [colab notebook](https://colab.research.google.com/drive/1ughLazoCppDFmUFDNuQcUcuHM0opGN4D) was used fine-tune the model and then downloaded in .md5 format.

### System Workflow
The workflow includes:
1. Trigger word detection.
2. Speech-to-text conversion.
3. Intent classification.
4. Response generation.
5. Text-to-speech output.

### Implementation Details
The system is built using various tools, including PyTorch for model development, Google APIs for speech recognition and synthesis, and LangChain for prompt engineering.

### Challenges and Solutions
Challenges faced during development included data collection, model training, and integration. Solutions involved using distinctive trigger words and modular development approaches to enhance system robustness.

### Future Work and Improvements
Potential enhancements include:
- Improved speech recognition for dialectal variations.
- Usage of the BERT model instead of the LSTM model for a more powerful trigger word detector.

### Conclusion
The Arabic Smart Assistant project successfully integrates machine learning and NLP to create a functional and user-friendly assistant for Arabic speakers, with plans for future enhancements to improve its capabilities.
