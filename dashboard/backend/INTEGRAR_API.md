
# Integração do histórico persistente à API VisionAI

Este módulo complementa a API atual. Não substitua o `api_server.py`
inteiro: preserve a câmera, o reconhecimento facial, o filtro temporal
e os endpoints de cadastro existentes.

## 1. Copie os arquivos

Copie `event_store.py` e `integrate_events.py` para a mesma pasta
em que está o `api_server.py` da AI BOX.

## 2. Faça a integração

Ative o ambiente virtual e execute:

```bash
cd /home/grupo13/eduvision
source .venv/bin/activate
python integrate_events.py
```

O script procura a estrutura esperada e cria `api_server.py.bak`
antes de gravar as alterações.

Se o script informar que não encontrou algum trecho, pare e leia
a mensagem. Não substitua o servidor inteiro por outro código.

## 3. Reinicie a API

```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 4. Valide os endpoints

Em outro terminal da AI BOX:

```bash
curl http://127.0.0.1:8000/api/events
curl http://127.0.0.1:8000/api/events/summary
```

O histórico pode começar vazio. Isso é esperado até a primeira
ocorrência confirmada.

## 5. Banco de dados

O banco é criado em:

`data/visionai_events.sqlite3`

Não apague esse arquivo se quiser preservar os registros.

## 6. Regra dos alertas

Uma ocorrência não autorizada precisa persistir por 4 segundos
para gerar um evento. O evento é resolvido depois de 3 segundos
sem pessoas não autorizadas.

Esta versão agrupa uma ocorrência contínua da cena. Ela não
identifica biometricamente uma pessoa ao longo de todos os frames.
Para separar pessoas desconhecidas simultâneas, será necessário
integrar IDs de rastreamento persistentes.
