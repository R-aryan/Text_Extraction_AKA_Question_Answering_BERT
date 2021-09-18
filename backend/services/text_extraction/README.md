# Text Extraction AKA Question Answering Using BERT

Performing Text Extraction also known as Question-Answering using BERT,and serving it Via REST API.


- More about BERT can be found [here](https://huggingface.co/bert-base-uncased)
- End to End NLP  Text Extraction Probelm.
- The Kaggle dataset can be found Here [Click Here](https://www.kaggle.com/c/tweet-sentiment-extraction/data)
- My kaggle Notebook can be found [here](https://www.kaggle.com/raryan/tweet-sentiment-extraction)
 
## Steps to Run the Project:
- create a virtual environment and install requirements.txt
  
### For Training
- After Setting up the environment go to [**backend/services/text_extraction/application/ai/training/**](https://github.com/R-aryan/Text_Extraction_AKA_Question_Answering_BERT/tree/main/backend/services/text_extraction/application/ai/training) and run **main.py** and the training will start.
- After training is complete the weights of the model will be saved in weights directory, and this weights can be used for inference.
  
### For Prediction/Inference
- Download the pre-trained weights from [here](https://drive.google.com/file/d/1uzDUH5J6kq9uQzgCIujlnphgRbhktIc1/view?usp=sharing) and place it inside the weights folder(**backend/services/text_extraction/application/ai/weights/trained_weights**)
- After setting up the environment: go to **backend/services/text_extraction/api** and run **app.py**.
- After running the above step the server will start.  
- You can send the POST request at this URL - **localhost:8080/text_extraction/api/v1/predict** (you can find the declaration of endpoint under **backend/services/text_extraction/api/__init__.py** )
- You can also see the logs under **(backend/services/text_extraction/logs)** directory.

### Following are the screenshots for the sample **request** and sample **response.**

- Request sample

![Sample request](https://github.com/R-aryan/Text_Extraction_AKA_Question_Answering_BERT/blob/main/msc/sample_request.png)
  <br>
  <br>
- Response Sample

![Sample response](https://github.com/R-aryan/Text_Extraction_AKA_Question_Answering_BERT/blob/main/msc/sample_response.png)
