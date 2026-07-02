FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "tensorflow-cpu>=2.16.0" "shap>=0.45.0"

COPY NVIDIA_Kursprognose_LSTM.ipynb _build_notebook.py ./

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
