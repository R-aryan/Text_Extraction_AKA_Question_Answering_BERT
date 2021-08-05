FROM python:3.7
RUN pip3 install --upgrade pip

ENV APP/app

RUN mkdir ${APP}
ENV PYTHONPATH="$PYTHONPATH:/app/Text_Extraction_AKA_Question_Answering_BERT"

COPY . ${APP}/Text_Extraction_AKA_Question_Answering_BERT/

WORKDIR ${APP}/Text_Extraction_AKA_Question_Answering_BERT/

RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["python","backend/services/text_extraction/api/app.py"]