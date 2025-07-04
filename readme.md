# MLFlow\_FastApi 

Proyecto que muestra cómo:

1. Registrar entrenamientos y métricas de *Machine Learning* con **MLflow**.
2. Exponer una pequeña API con **FastAPI** 

---

##  Resumen

- **Objetivo**: aprender los fundamentos de MLflow y FastAPI en un mismo repositorio.
- **Datos**: Dataset de *Diabetes* de *scikit‑learn*.
- **Modelos**: se entrenan brevemente Random Forest, XGBoost y K‑Nearest Neighbors.
- **API**: incluye endpoints de saludo, cálculo simple, traducción (es→en) y *question‑answering*.

---

##  Estructura mínima

```
MLFlow_FastApi/
├── MLFLOW/        # Scripts de entrenamiento + tracking
│   └── main.py
├── FASTAPI/       # Aplicación web
│   └── main.py
├── mlruns/        # Resultados de MLflow (se crea al entrenar)
└── README.md      # Este archivo 
```

---

## 🛠️ Requisitos rápidos

```bash
pip install -r MLFLOW/requirementes.txt   # MLflow + ciencia de datos
pip install -r FASTAPI/requirementes.txt  # API y modelos HF
```

Python ≥ 3.10 recomendado.

---

##  Puesta en marcha

1. **Entrenar y registrar** modelos (opcional, \~1 min):
   ```bash
   python MLFLOW/main.py
   ```
2. **Levantar la API**:
   ```bash
   fastapi run main.py --reload
   ```
3. Explora los endpoints en `http://localhost:8000/docs` (Swagger UI).

---

##  Ejemplo rápido

```bash
# Traducir "Hola mundo" a inglés
curl "http://localhost:8000/traduce?texto=Hola%20mundo"
```

Respuesta esperada:

```json
{"traduccion": "Hello world"}
```


> Autor: Nauzet Fernández Lorenzo

