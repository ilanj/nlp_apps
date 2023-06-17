import os


# FastAPI imports
from fastapi import FastAPI, Request
# uvicorn
import uvicorn
from pydantic import BaseModel
# spacy
import spacy

nlp = spacy.load('en_core_web_trf')

app = FastAPI()

# HTTP_URL = COMMON_ATTRIBUTES['HTTP_URL']
# HTTP_STATUS_CODE = COMMON_ATTRIBUTES['HTTP_STATUS_CODE']


class RequestModel(BaseModel):
    request_string: str = None


@app.post("/extract")
async def root(request_model: RequestModel):
    '''

    :param request_model: string (can view from swagger)
    :return: dict of entities
    '''
    doc = nlp(request_model.request_string)
    results = list()
    for ent in doc.ents:
        results.append((ent.lemma_, ent.label_))
        print(ent)
    response = dict(results)
    properties = {'custom_dimensions': response}
    results.clear()
    return response


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=5001, log_level="info")
