from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from datetime import datetime

app = FastAPI()


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
qa = pipeline("question-answering")


class QAIn(BaseModel):
    contexto: str
    pregunta: str


@app.get("/")
def root():
    """Ping básico con sello de tiempo."""
    ahora = datetime.now().isoformat(timespec="seconds")
    return {"message": "API activa", "timestamp": ahora}


@app.get("/saludo")
def saludo(nombre: str = "usuario", mayusculas: bool = False):
    """
    Saluda y opcionalmente grita (mayúsculas).
    Extra: añade la hora local para darle un toquecito simpático.
    """
    saludo = f"¡Hola, {nombre}! Son las {datetime.now().strftime('%H:%M')}."
    return {"saludo": saludo.upper() if mayusculas else saludo}


@app.get("/cuadrado")
def cuadrado(num: float):
    """Devuelve el número y su cuadrado."""
    return {"numero": num, "cuadrado": num ** 2}


@app.get("/traduce")
def traduce(texto: str):
    """
    Traduce de español a inglés (ES/EN)
    """
    result = translator(texto, max_length=200)[0]
    return {"traduccion": result["translation_text"]}


@app.post("/respuesta")
def respuesta(data: QAIn):
    """
    Responde preguntas dado un contexto usando QA pipeline.
    """
    result = qa(question=data.pregunta, context=data.contexto)
    return {
        "answer": result["answer"],
        "score": round(result["score"], 2),
        "start": result["start"],
        "end": result["end"]
    }
